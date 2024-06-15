import numpy as np
import torch
import torch.nn.functional as F
from torch import jit, nn
from torch.nn.utils import spectral_norm

from distributions import SquashedNormal, TruncatedNormal
from utils import weight_init


def maybe_sn(m, use_sn):
    return spectral_norm(m) if use_sn else m


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, spectral_norms):
        super().__init__()
        assert len(hidden_dims) == len(spectral_norms)
        layers = []
        for dim, use_sn in zip(hidden_dims, spectral_norms):
            layers += [
                maybe_sn(nn.Linear(input_dim, dim), use_sn),
                nn.ReLU(inplace=True),
            ]
            input_dim = dim

        layers += [nn.Linear(input_dim, output_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DoubleQCritic(nn.Module):
    def __init__(
        self,
        obs_type,
        obs_dim,
        action_dim,
        feature_dim,
        hidden_dims,
        spectral_norms,
        use_ln,
    ):
        super().__init__()

        self.obs_type = obs_type

        if obs_type == "pixels":
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
            )
            trunk_dim = feature_dim + action_dim
        elif use_ln:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dims[0]),
                nn.LayerNorm(hidden_dims[0]),
                nn.Tanh(),
            )
            trunk_dim = hidden_dims[0]
        else:
            self.trunk = nn.Identity()
            trunk_dim = obs_dim + action_dim

        self.q1_net = MLP(trunk_dim, 1, hidden_dims, spectral_norms)
        self.q2_net = MLP(trunk_dim, 1, hidden_dims, spectral_norms)

        self.apply(weight_init)

    def forward(self, obs, action):
        inpt = obs if self.obs_type == "pixels" else torch.cat([obs, action], dim=-1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == "pixels" else h

        q1 = self.q1_net(h.to(dtype=torch.float32))
        q2 = self.q2_net(h.to(dtype=torch.float32))

        return q1, q2


class DeterministicActor(nn.Module):
    def __init__(
        self, obs_dim, action_dim, feature_dim, hidden_dims, spectral_norms, use_ln
    ):
        super().__init__()

        self.policy_in_dim = obs_dim
        if use_ln:
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
            )
            self.policy_in_dim = feature_dim
        else:
            self.trunk = nn.Identity()

        self.policy_net = MLP(
            self.policy_in_dim, action_dim, hidden_dims, spectral_norms
        )

        self.apply(weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)
        mu = self.policy_net(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)
        return dist


class StochasticActor(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        feature_dim,
        hidden_dims,
        spectral_norms,
        log_std_bounds,
        use_ln,
    ):
        super().__init__()

        self.log_std_bounds = log_std_bounds

        self.policy_in_dim = obs_dim
        if use_ln:
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
            )
            self.policy_in_dim = feature_dim
        else:
            self.trunk = nn.Identity()

        self.policy_net = MLP(
            self.policy_in_dim, 2 * action_dim, hidden_dims, spectral_norms
        )

        self.apply(weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        mu, log_std = self.policy_net(h.to(dtype=torch.float32)).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        return dist


class Encoder(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()

        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_dim, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.apply(weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class DQNCritic(nn.Module):
    def __init__(self, repr_dim, num_actions, hidden_dim, feature_dim, trunk_type):
        super().__init__()

        # Create Trunk
        if trunk_type == "id":
            self.trunk = nn.Identity()
            self.feature_dim = repr_dim
        elif trunk_type == "proj":
            self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim))
            self.feature_dim = feature_dim
        elif trunk_type == "proj+ln+tanh":
            self.trunk = nn.Sequential(
                nn.Linear(repr_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.Tanh(),
            )
            self.feature_dim = feature_dim

        # Critic Heads
        self.V = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.A = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        self.apply(weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        v = self.V(h)
        a = self.A(h)
        q = v + a - a.mean(1, keepdim=True)
        return q


class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim,  # (s, a) |s| + |a| (i.e. 30)
        hidden_dim,
        enc_input_dim=None,
        enc_output_dim=None,
        output_dim=1,
    ):
        super().__init__()

        if enc_input_dim is not None:
            if not enc_output_dim:
                enc_output_dim = input_dim
            self.encoder = Encoder(enc_input_dim)
            self.encoder_trunk = nn.Sequential(
                nn.Linear(self.encoder.repr_dim, enc_output_dim),
                nn.LayerNorm(enc_output_dim),
                nn.Tanh(),
            )
        else:
            self.encoder = nn.Identity()
            self.encoder_trunk = nn.Identity()

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

        self.apply(weight_init)

    def encode(self, obs):
        return self.encoder_trunk(self.encoder(obs))

    def forward(self, obs, next_obs=None, act=None, encode=False):
        if encode:
            obs = self.encode(obs)
        if not act:
            h = obs
        else:
            h = torch.cat([obs, act], dim=1)

        return self.trunk(h.to(dtype=torch.float32))
