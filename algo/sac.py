import torch
import torch.nn.functional as F

import utils
from nets import DoubleQCritic, StochasticActor


class SACAgent:
    """Soft Actor-Critic."""

    def __init__(
        self,
        name,
        obs_type,
        obs_shape,
        action_dim,
        action_range,
        device,
        lr,
        nstep,
        batch_size,
        log_std_bounds,
        critic_target_tau,
        critic_use_ln,
        critic_hidden_dims,
        critic_spectral_norms,
        actor_hidden_dims,
        feature_dim,
        actor_use_ln,
        actor_spectral_norms,
        num_expl_steps,
        init_temperature,
        update_every_steps,
        use_tb,
    ):

        self.device = device
        self.name = name
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.action_range = action_range

        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.batch_size = batch_size
        self.lr = lr
        self.nstep = nstep

        self.feature_dim = feature_dim
        self.actor_hidden_dims = actor_hidden_dims
        self.actor_spectral_norms = actor_spectral_norms
        self.log_std_bounds = log_std_bounds

        self.actor_use_ln = actor_use_ln
        self.critic_use_ln = critic_use_ln

        # models
        self.actor = StochasticActor(
            obs_shape[0],
            action_dim,
            feature_dim,
            actor_hidden_dims,
            actor_spectral_norms,
            log_std_bounds,
            actor_use_ln,
        ).to(device)

        self.eval_actor = StochasticActor(
            obs_shape[0],
            action_dim,
            feature_dim,
            actor_hidden_dims,
            actor_spectral_norms,
            log_std_bounds,
            actor_use_ln,
        ).to(device)

        self.eval_actor.load_state_dict(self.actor.state_dict())

        self.critic = DoubleQCritic(
            obs_type,
            obs_shape[0],
            action_dim,
            feature_dim,
            critic_hidden_dims,
            critic_spectral_norms,
            critic_use_ln,
        ).to(device)
        self.critic_target = DoubleQCritic(
            obs_type,
            obs_shape[0],
            action_dim,
            feature_dim,
            critic_hidden_dims,
            critic_spectral_norms,
            critic_use_ln,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # hparams
        self.init_temperature = init_temperature
        self.log_alpha = torch.tensor(init_temperature, dtype=torch.float64).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        dist = self.actor(obs)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        action = action.clamp(*self.action_range)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        # compute target and detach
        dist = self.actor(next_obs)
        next_action = dist.sample()
        next_log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * next_log_prob
        target_Q = reward + (discount * target_V)
        target_Q = target_Q.detach()

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        return metrics

    def reset_noise(self):
        self.log_alpha = torch.tensor(self.init_temperature, dtype=torch.float64).to(
            self.device
        )
        self.log_alpha.requires_grad = True
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr)

    def reinit_policy(self):
        """Reinitializes the actor optimizer's state."""

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

    def update_actor_and_alpha(self, obs, step):
        metrics = dict()

        utils.set_requires_grad(self.critic, False)

        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = (self.alpha.detach() * log_prob - Q).mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()

        # optimize alpha
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        utils.set_requires_grad(self.critic, True)

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            # metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            metrics["actor_ent"] = self.alpha * (-log_prob).mean()
            metrics["actor_alpha"] = self.alpha.item()
            metrics["actor_alpha_loss"] = alpha_loss.item()

        return metrics

    def update(self, batch, step):

        if len(batch) == 5:
            obs, action, reward, discount, next_obs = batch
        else:
            obs, action, reward, discount, next_obs = batch[: -2 * (self.nstep - 1)]

        metrics = dict()

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step)
        )

        # update actor
        metrics.update(self.update_actor_and_alpha(obs.detach(), step))

        # update critic target
        if step % self.update_every_steps == 0:
            utils.soft_update_params(
                self.critic, self.critic_target, self.critic_target_tau
            )

        return metrics
