import numpy as np
import torch
import torch.nn.functional as F

import utils
from nets import Discriminator

from .agent import Agent


class DACAgent(Agent):
    """Discriminator actor-critic baseline agent.

    Paper: https://arxiv.org/abs/1809.02925.
    """

    def __init__(
        self,
        name,
        batch_size,
        task,
        device,
        expert_dir,
        num_demos,
        feature_dim,
        reward_mode,
        algo,
        representation,
        disc_hidden_dim,
        disc_type,
        divergence,
        num_policy_updates_per_disc_update,
    ):

        super().__init__(name, task, device, algo)
        assert disc_type == "s" or disc_type == "ss" or disc_type == "sa"
        self.disc_type = disc_type  # r(s) or r(s, s')
        self.divergence = divergence

        # demos_path = expert_dir + task + "/expert_demos.pkl"
        self.representation = representation

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
            self.discriminator.parameters(), lr=self.policy.lr
        )

        self.reward_mode = reward_mode
        self.device = device
        self.batch_size = batch_size
        self.discriminator_list = []

        self.num_policy_updates_per_disc_update = num_policy_updates_per_disc_update
        self.counter = 0

        self.train()

    def train(self):
        self.policy.train()
        self.discriminator.train()

    def __repr__(self):
        return self.reward_mode

    def encode(self, obs):
        if self.representation == "rl_encoder":
            return self.policy.actor.trunk(self.policy.encoder(obs))
        elif self.representation == "discriminator":
            return self.discriminator.encode(obs)

    def compute_online_rewards(self, obs, step, fictitious=False):
        with torch.no_grad():
            with utils.eval_mode(self.discriminator):
                d = self.discriminator(obs)
                rewards = d

            return rewards.flatten().detach().reshape(-1, 1)

    def compute_offline_rewards(self, time_steps, episode):
        traj = np.stack([time_step.observation for time_step in time_steps], 0)
        action_traj = np.stack([time_step.action for time_step in time_steps], 0)

        obs = torch.tensor(traj[:-1]).to(self.device)
        if self.policy.obs_type == "pixels":
            obs = self.encode(obs)

        if self.disc_type == "s":
            obs = torch.tensor(traj[:-1]).to(self.device)
        if self.disc_type == "sa":
            actions = torch.tensor(action_traj[1:]).to(self.device)
            obs = torch.cat([obs, actions], dim=1).to(self.device)
        else:
            # Concatenate state and next state
            next_obs = torch.tensor(traj[1:]).to(self.device)
            if self.policy.obs_type == "pixels":
                next_obs = self.encode(next_obs)

            obs = torch.cat([obs, next_obs], dim=1).to(self.device)
        rewards = self.compute_online_rewards(obs, episode * obs.shape[0]).cpu().numpy()

        # Preprocess time steps
        for i in range(len(time_steps) - 1):
            time_steps[i] = time_steps[i]._replace(reward=rewards[i, 0])
        return time_steps, np.sum(rewards)

    def update_discriminator(self, batch, expert_loader, expert_iter, step):
        metrics = dict()

        # Grab Expert Data
        expert_batch = next(expert_iter)
        if len(batch) > 3:
            expert_obs, expert_actions, expert_obs_next = utils.to_torch(
                expert_batch[:3], self.device
            )
        else:
            expert_obs, expert_actions, expert_obs_next = utils.to_torch(
                expert_batch, self.device
            )
        bs = (
            self.batch_size // 2
            if self.policy.obs_type == "states"
            else expert_obs.size(0)
        )  # set batch size for state vs. vision (hacky for now but should work I think)

        expert_obs = expert_obs.float()[:bs]  # won't change in vision construct
        expert_actions = np.squeeze(expert_actions, axis=1)[
            :bs
        ]  # won't change in vision construct

        # Grab Policy Data
        obs = batch[0]
        actions = batch[1]
        policy_obs = obs[:bs].to(self.device)
        policy_actions = np.squeeze(actions, axis=1)[:bs].to(self.device)

        with torch.no_grad():
            # first encode all images
            if self.policy.obs_type == "pixels":
                expert_obs = self.encode(expert_obs)
                policy_obs = self.encode(policy_obs)

            # now handle specific discriminator input instances
            if self.disc_type == "ss":
                # (s, s')
                expert_obs_next = expert_obs_next.float()[:bs]
                obs_next = batch[-1]
                policy_obs_next = obs_next[:bs]
                if self.policy.obs_type == "pixels":
                    expert_obs_next = self.encode(expert_obs_next)  # (B, feature_dim)
                    policy_obs_next = self.encode(policy_obs_next)  # (B, feature_dim)

                expert_data = torch.cat([expert_obs, expert_obs_next], dim=1).to(
                    self.device
                )  # (B, 2 * feature_dim)
                policy_data = torch.cat([policy_obs, policy_obs_next], dim=1).to(
                    self.device
                )  # (B, 2 * feature_dim)
            elif self.disc_type == "sa":
                expert_data = torch.cat([expert_obs, expert_actions], dim=1).to(
                    self.device
                )  # (B, feature_dim + action_dim)
                policy_data = torch.cat([policy_obs, policy_actions], dim=1).to(
                    self.device
                )  # (B, feature_dim + action_dim)
            elif self.disc_type == "s":
                expert_data = expert_obs  # (B, feature_dim)
                policy_data = policy_obs  # (B, feature_dim)
            else:
                raise NotImplementedError(f"{self.disc_type} input not supported")

            # get the full input (expert input + policy input)
            disc_input = torch.cat(
                [expert_data, policy_data], dim=0
            )  # (2 * B, feature_dim_of_interest) for imgs, (2 * B, state_dim) for states

        # discriminator output
        disc_output = self.discriminator(disc_input, encode=False)  # (2 * B, 1)
        assert disc_output.size(0) == 2 * bs, "wrong batch size!"

        # compute loss based off of divergence measure
        if self.divergence == "js":
            ones = torch.ones(bs, device=self.device)
            zeros = torch.zeros(bs, device=self.device)
            disc_label = torch.cat([ones, zeros]).unsqueeze(dim=1)  # (2 * B, 1)
            # add constraint here
            dac_loss = F.binary_cross_entropy_with_logits(
                disc_output, disc_label, reduction="sum"
            )
            dac_loss /= bs
        elif self.divergence == "rkl":
            disc_expert, disc_policy = torch.split(disc_output, bs, dim=0)
            dac_loss = torch.mean(torch.exp(-disc_expert) + disc_policy)
        elif self.divergence == "wass":
            disc_expert, disc_policy = torch.split(disc_output, bs, dim=0)
            dac_loss = torch.mean(disc_policy - disc_expert)

        # dis_expert = self.discriminator(expert_traj,encode=False)
        # dis_policy = self.discriminator(policy_traj,encode=False)

        # dac_loss = torch.mean(
        #    self.discriminator(expert_data, encode=False)
        # ) - torch.mean(self.discriminator(policy_data, encode=False))
        metrics["train/disc_loss"] = dac_loss.mean().item()

        # expert_obs, policy_obs = torch.split(disc_input, batch_size, dim=0)

        grad_pen = utils.compute_gradient_penalty(
            self.discriminator, expert_data, policy_data
        )  # used to detach
        grad_pen /= policy_obs.shape[0]

        # calculate divergence
        # sample = expert_loader.dataset
        # expert_traj_obs = torch.stack(list(utils.to_torch(sample.obs,self.device)), dim=0)
        # expert_traj_actions = torch.stack(list(utils.to_torch(sample.act.squeeze(),self.device)), dim=0)
        # expert_traj = torch.cat([expert_traj_obs, expert_traj_actions], dim=2)
        # policy_traj = torch.cat([obs, actions], dim=len(obs.shape)-1)
        # metrics["divergence"] = torch.mean(self.discriminator(expert_traj,encode=False)) - torch.mean(self.discriminator(policy_traj,encode=False))

        self.discriminator_opt.zero_grad(set_to_none=True)
        dac_loss.backward()
        grad_pen.backward()
        self.discriminator_opt.step()

        return metrics

    def compute_divergence(self, expert_loader, on_policy_data):
        obs = on_policy_data[0]
        actions = on_policy_data[1]
        sample = expert_loader.dataset

        expert_traj_obs = torch.stack(
            list(utils.to_torch(sample.obs, self.device)), dim=0
        )
        expert_traj_actions = torch.stack(
            list(utils.to_torch(sample.act, self.device)), dim=0
        )

        # if given pixels, encode them before passing to discriminator
        if self.policy.obs_type == "pixels":
            T = expert_traj_obs.size(0)
            expert_traj_obs = utils.flatten_seq(expert_traj_obs)
            expert_traj_obs = self.encode(expert_traj_obs)
            expert_traj_obs = utils.unflatten_seq(expert_traj_obs, T)

            T = obs.size(0)
            obs = utils.flatten_seq(obs)
            obs = self.encode(obs)
            obs = utils.unflatten_seq(obs, T)

            print(f"new expert obs shape: {expert_traj_obs.size()}")
            print(f"new obs shape: {obs.size()}")

        expert_traj = torch.cat([expert_traj_obs, expert_traj_actions], dim=2)
        policy_traj = torch.cat([obs, actions], dim=2)
        return torch.mean(self.discriminator(expert_traj, encode=False)) - torch.mean(
            self.discriminator(policy_traj, encode=False)
        )

    def update(self, env, buffer, replay_iter, expert_loader, expert_iter, step):

        metrics = dict()
        if step % self.policy.update_every_steps != 0:
            return metrics

        assert repr(self) != "online_imitation"

        batch = next(replay_iter)
        batch = utils.to_torch(batch, self.device)

        # Discriminator Update (update every X times)
        if self.counter % self.num_policy_updates_per_disc_update == 0:
            metrics.update(
                self.update_discriminator(batch, expert_loader, expert_iter, step)
            )

        # Policy Update
        metrics.update(self.policy.update(batch, step))

        # increment counter
        self.counter += 1

        return metrics

    # if __name__ == '__main__':
