import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from nets import DeterministicActor, DoubleQCritic, Encoder


class DDPGAgent:
    """Deep deterministic policy gradient. This is used for DrQv2 tasks."""

    def __init__(
        self,
        name,
        obs_type,
        obs_shape,
        action_dim,
        device,
        lr,
        nstep,
        batch_size,
        critic_target_tau,
        num_expl_steps,
        critic_use_ln,
        critic_hidden_dims,
        critic_spectral_norms,
        actor_use_ln,
        actor_hidden_dims,
        feature_dim,
        actor_spectral_norms,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        use_tb,
    ):
        self.name = name
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.feature_dim = feature_dim
        self.obs_type = obs_type
        self.lr = lr
        self.nstep = nstep
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.internal_step = 0

        self.actor_use_ln = actor_use_ln
        self.critic_use_ln = critic_use_ln

        # needed by GAIL
        self.batch_size = batch_size

        # models
        if obs_type == "pixels":
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(obs_shape[0]).to(device)
            self.obs_dim = self.encoder.repr_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0]

        self.actor = DeterministicActor(
            self.obs_dim,
            action_dim,
            feature_dim,
            actor_hidden_dims,
            actor_spectral_norms,
            actor_use_ln,
        ).to(device)

        self.eval_actor = DeterministicActor(
            self.obs_dim,
            action_dim,
            feature_dim,
            actor_hidden_dims,
            actor_spectral_norms,
            actor_use_ln,
        ).to(device)

        self.eval_actor.load_state_dict(self.actor.state_dict())

        self.critic = DoubleQCritic(
            obs_type,
            self.obs_dim,
            action_dim,
            feature_dim,
            critic_hidden_dims,
            critic_spectral_norms,
            critic_use_ln,
        ).to(device)
        self.critic_target = DoubleQCritic(
            obs_type,
            self.obs_dim,
            action_dim,
            feature_dim,
            critic_hidden_dims,
            critic_spectral_norms,
            critic_use_ln,
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        if obs_type == "pixels":
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        else:
            self.encoder_opt = None

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.encoder.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        stddev = utils.schedule(self.stddev_schedule, self.internal_step)
        h = self.encoder(obs)
        dist = self.actor(h, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, self.internal_step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)

        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        utils.set_requires_grad(self.critic, False)

        stddev = utils.schedule(self.stddev_schedule, self.internal_step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        utils.set_requires_grad(self.critic, True)

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, batch, step):
        # pri_time = time.time()

        metrics = dict()

        # if step % self.update_every_steps != 0:
        #     return metrics
        # start_time = time.time()
        # obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)
        obs, action, reward, discount, next_obs = batch
        # print("batch time", time.time()-start_time)

        # start_time = time.time()
        obs = self.aug_and_encode(obs)
        # print("obs aug and encode", time.time()-start_time)
        # start_time = time.time()
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)
        # print("next obs aug and encode", time.time()-start_time)

        # start_time = time.time()
        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()
        # print("log", time.time()-start_time)

        # update critic
        # start_time = time.time()
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step)
        )
        # print("update critic", time.time()-start_time)

        # start_time = time.time()
        # update actor
        metrics.update(self.update_actor(obs.detach(), step))
        # print("update actor", time.time()-start_time)

        # start_time = time.time()
        # update critic target
        with torch.no_grad():
            utils.soft_update_params(
                self.critic, self.critic_target, self.critic_target_tau
            )
        # print("update critic target", time.time()-start_time)
        # print("total inner update time", time.time()-pri_time)

        return metrics

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

    def reset_step(self):
        self.internal_step = 0

    def reset_step(self):
        self.internal_step = 0
