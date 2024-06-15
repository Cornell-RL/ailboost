from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

import utils
from nets import Discriminator

from .agent import Agent


class BoostingAgent(Agent):
    def __init__(
        self,
        name,
        batch_size,
        task,
        device,
        feature_dim,
        algo,
        representation,
        disc_hidden_dim,
        disc_type,
        disc_update_iter,
        n_learners,
        discount,
        divergence,
        mixin_schdl,
    ):

        super().__init__(name, task, device, algo)
        assert disc_type == "s" or disc_type == "ss" or disc_type == "sa"
        self.disc_type = disc_type  # r(s), r(s, s'), r(s, a)
        self.algo = algo

        self.divergence = divergence
        self.representation = representation

        print(f"max number of learners: {n_learners}")

        assert mixin_schdl.startswith("constant") or mixin_schdl.startswith(
            "linear"
        ), "others not implemented"

        if mixin_schdl.startswith("constant"):
            schdl, ratio = mixin_schdl.split("_")
            ratio = float(ratio)
            assert (
                0.5 <= ratio <= 1.0
            ), "can't use less replay samples to train I guess for now"
            self.ratio = ratio
        else:
            schdl, min_ratio, max_ratio, duration = mixin_schdl.split("_")
            self.min_ratio = min_ratio
            self.max_ratio = max_ratio
            self.duration = duration
        self.schdl = schdl

        if self.representation == "rl_encoder":
            self.discriminator = Discriminator(feature_dim, disc_hidden_dim).to(device)
        elif self.representation == "discriminator":
            enc_in_dim = (
                self.policy.obs_shape[0]
                if disc_type == "s"
                else 2 * self.policy.obs_shape[0]
            )
            self.discriminator = Discriminator(
                feature_dim, disc_hidden_dim, enc_in_dim
            ).to(device)

        self.discriminator_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.policy.lr, maximize=False
        )

        self.disc_update_iter = disc_update_iter
        self.device = device
        self.batch_size = batch_size
        self.learners = deque(maxlen=n_learners)
        self.discount = discount

        self.train()

    def train(self):
        self.policy.train()
        self.discriminator.train()

    def __repr__(self):
        return "boosting"

    def replay_to_total_ratio(self, step):
        if self.schdl == "constant":
            return self.ratio
        else:
            mix = np.clip(step / self.duration, 0.0, 1.0)
            return (1.0 - mix) * self.min_ratio + mix * self.max_ratio

    def num_expert_samples(self, replay_bs, ratio):
        """1/ratio = total/replay = (replay + expert)/replay = 1 + expert/replay"""
        inv_ratio = 1.0 / ratio  # should be defined as ratio is not 0
        expert_to_replay_ratio = inv_ratio - 1.0
        return int(expert_to_replay_ratio * replay_bs)

    def encode(self, obs):
        if self.representation == "rl_encoder":
            return self.policy.actor.trunk(self.policy.encoder(obs))
        elif self.representation == "discriminator":
            return self.discriminator.encode(obs)

    def get_rewards(self, obs):
        if self.policy.obs_type == "pixels":
            obs = self.encode(obs)
        with torch.no_grad(), utils.eval_mode(self.discriminator):
            d = self.discriminator(obs)
        return d.flatten().detach().reshape(-1, 1)

    def reset_policy(self, reinit_policy=False):
        self.policy.reset_noise()
        if reinit_policy:
            self.policy.reinit_policy()

    def add_learner(self):
        self.learners.append(self.policy.actor.state_dict())

    def sample_learner(self, weights):
        """Returns a policy from ensemble of policies"""
        if len(self.learners) == 0:
            return

        sampled_weights = np.random.choice(self.learners, p=weights)
        self.policy.eval_actor.load_state_dict(sampled_weights)

    @torch.no_grad()
    def boosted_act(self, obs):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        if self.algo.name == "ddpg":
            stddev = utils.schedule(
                self.policy.stddev_schedule, self.policy.internal_step
            )
            dist = self.policy.eval_actor(obs, stddev)
        else:
            dist = self.policy.eval_actor(obs)
        action = dist.mean
        return action.cpu().numpy()[0]

    def update_discriminator(self, replay_iter, expert_iter):
        metrics = dict()

        for _ in range(self.disc_update_iter):
            policy_batch = next(replay_iter)
            expert_batch = next(expert_iter)

            # expert_data = np.concatenate([expert_batch[0], np.squeeze(expert_batch[1], axis=1)], axis=1)
            # policy_data = np.concatenate([policy_batch[0], np.squeeze(policy_batch[1], axis=1)], axis=1)

            expert_data = torch.cat(
                [expert_batch[0], torch.squeeze(expert_batch[1], dim=1)], dim=1
            )

            policy_data = np.concatenate([policy_batch[0], policy_batch[1]], axis=1)
            policy_data = torch.from_numpy(policy_data)

            # batch
            batch_size = self.batch_size // 2
            expert_data = expert_data[:batch_size]
            policy_data = policy_data[:batch_size]

            expert_data, policy_data = utils.to_torch(
                (expert_data, policy_data), self.device
            )
            disc_input = torch.cat([expert_data, policy_data], dim=0)
            disc_output = self.discriminator(disc_input, encode=False)

            if self.divergence == "js":
                ones = torch.ones(batch_size, device=self.device)
                zeros = torch.zeros(batch_size, device=self.device)
                disc_label = torch.cat([ones, zeros]).unsqueeze(dim=1)
                # add constraint here
                dac_loss = F.binary_cross_entropy_with_logits(
                    disc_output, disc_label, reduction="sum"
                )
                dac_loss /= batch_size
            elif self.divergence == "rkl":
                disc_expert, disc_policy = torch.split(disc_output, batch_size, dim=0)
                dac_loss = torch.mean(torch.exp(-disc_expert) + disc_policy)
            elif self.divergence == "wass":
                disc_expert, disc_policy = torch.split(disc_output, batch_size, dim=0)
                dac_loss = torch.mean(disc_policy - disc_expert)

            metrics["train/disc_loss"] = dac_loss.mean().item()

            grad_pen = utils.compute_gradient_penalty(
                self.discriminator, expert_data, policy_data
            )
            grad_pen /= batch_size

            self.discriminator_opt.zero_grad(set_to_none=True)
            dac_loss.backward()
            grad_pen.backward()
            self.discriminator_opt.step()

        return metrics

    @torch.no_grad()
    def compute_divergence(self, expert_loader, on_policy_data):
        obs = on_policy_data[0]
        actions = on_policy_data[1]

        # expert stuff
        sample = expert_loader.dataset
        expert_traj_obs = torch.stack(
            list(utils.to_torch(sample.obs, self.device)), dim=0
        )
        expert_traj_actions = torch.stack(
            list(utils.to_torch(sample.act, self.device)), dim=0
        )
        expert_traj = torch.cat([expert_traj_obs, expert_traj_actions], dim=2)

        # policy stuff
        policy_traj = torch.cat([obs, actions], dim=len(obs.shape) - 1)

        if self.divergence == "wass":
            return torch.mean(
                self.discriminator(expert_traj, encode=False)
            ) - torch.mean(self.discriminator(policy_traj, encode=False))
        elif self.divergence == "rkl":
            disc_expert = self.discriminator(expert_traj)
            disc_out = torch.exp(-disc_expert).mean()
            policy_out = self.discriminator(policy_traj).mean()

            return disc_out + policy_out
        else:
            raise NotImplementedError

    def update(self, replay_iter, expert_iter, step):

        metrics = dict()
        if step % self.policy.update_every_steps != 0:
            return metrics

        replay_batch = next(replay_iter)
        expert_batch = next(expert_iter)

        # For the actions: TODO: fix
        expert_batch[1] = torch.squeeze(expert_batch[1], dim=1)

        replay_batch = utils.to_torch(replay_batch, self.device)
        expert_batch = utils.to_torch(expert_batch, self.device)
        ratio = self.replay_to_total_ratio(step)
        expert_samples = self.num_expert_samples(replay_batch[0].size(0), ratio)

        # get appropriate amt of stuff in expert batch and concatenate
        expert_batch = utils.torch_map(lambda x: x[:expert_samples], expert_batch)
        state = torch.cat([replay_batch[0], expert_batch[0]], dim=0)
        action = torch.cat([replay_batch[1], expert_batch[1]], dim=0)
        next_state = torch.cat([replay_batch[4], expert_batch[2]], dim=0)

        ratio = int(1 / self.replay_to_total_ratio(step))
        # total_samples = state.size(0)
        discount = replay_batch[3].tile(
            (ratio, 1)
        )  # maybe take first 'total_samples' to avoid batching bug

        # set batch to appropriate stuff
        batch = [state, action, None, discount, next_state]

        # Get the current reward estimate
        # TODO: Handle different disc input types
        rewards = self.get_rewards(torch.cat([batch[0], batch[1]], dim=1))

        # Nstep calculation
        if self.policy.nstep > 1:
            n_int = self.policy.nstep - 1

            int_obs = replay_batch[-2 * n_int : -n_int]
            int_expert_obs = expert_batch[-2 * n_int : -n_int]

            int_act = replay_batch[-n_int:]
            int_expert_act = expert_batch[-n_int:]

            int_batch = []
            for o, a, eo, ea in zip(int_obs, int_act, int_expert_obs, int_expert_act):
                obs = torch.cat([o, eo], dim=0)
                act = torch.cat([a, ea], dim=0)
                int_batch.append(torch.cat([obs, act], dim=1))

            for b in int_batch:
                int_r = self.get_rewards(b)
                rewards += self.discount * int_r

        # set the rewards
        batch[2] = rewards

        # update the policy
        metrics = self.policy.update(batch, step)

        return metrics
