import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.lerp_(param.data, tau)


def set_requires_grad(net, value):
    for param in net.parameters():
        param.requires_grad_(value)


def to_torch(xs, device, dtype=None):
    if dtype is not None:
        xs = (x.astype(dtype) for x in xs)
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def torch_map(fn, xs):
    """For non np-to-torch conversion functions."""
    return tuple(fn(x) for x in xs)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


def compute_gradient_penalty(discriminator, expert_data, policy_data):
    alpha = torch.rand(expert_data.size(0), 1)

    alpha = alpha.expand_as(expert_data).to(expert_data.device)

    mixup_data = alpha * expert_data + (1 - alpha) * policy_data
    mixup_data.requires_grad = True

    disc = discriminator(mixup_data)
    ones = torch.ones(disc.size()).to(disc.device)
    grad = autograd.grad(
        outputs=disc,
        inputs=mixup_data,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad_pen = 10 * (grad.norm(2, dim=1) - 1).pow(2).sum()
    return grad_pen


import pickle
from collections import defaultdict
from pathlib import Path

import dm_env

# data collection utils (should collect for both state and vision-based observations)
from agent.agent import Agent
from suite.dmc_multiobs import make as make_multiobs


def collect_data(agent: Agent, env: dm_env.Environment, n_episodes: int):
    """Collects data from specified agent."""
    trajs = defaultdict(list)
    rews = []

    for _ in range(n_episodes):
        ts = env.reset()
        ep_rew = 0.0

        ep_states = []
        ep_actions = []
        ep_pixels = []
        ep_next_states = []
        ep_next_pixels = []
        ep_rewards = []
        ep_dones = []
        ep_discounts = []

        while not ts.last():
            ob = ts.observation

            with torch.no_grad():
                state = ob["state"].astype(np.float32)
                action = agent.act(
                    state, 0, eval_mode=True
                )  # step doesn't matter when doing eval so just set it to 0

            # log
            ep_states.append(ob["state"])
            ep_pixels.append(ob["pixels"])
            ep_actions.append(action)

            # step env (guaranteed to not be first, so reward should always be defined)
            ts = env.step(action)
            r = ts.reward
            ep_rew += r

            # log next state things
            n_ob = ts.observation
            ep_next_states.append(n_ob["state"])
            ep_next_pixels.append(n_ob["pixels"])
            ep_rewards.append(r)
            ep_dones.append(bool(ts.last() == True))
            ep_discounts.append(
                ts.discount
            )  # this is what is gonna discount the next state (0 if last, which is what we want)

        trajs["states"].append(np.stack(ep_states))
        trajs["actions"].append(np.stack(ep_actions))
        trajs["pixels"].append(np.stack(ep_pixels))
        trajs["next_states"].append(np.stack(ep_next_states))
        trajs["next_pixels"].append(np.stack(ep_next_pixels))
        trajs["rewards"].append(np.stack(ep_rewards))
        trajs["discount"].append(np.stack(ep_discounts))
        trajs["dones"].append(np.stack(ep_dones))

        rews.append(ep_rew)

    print(
        f"avg episode reward for this agent across {n_episodes} episodes: {np.mean(rews)}"
    )
    return trajs, rews  # dict of lists of trajectory information


if __name__ == "__main__":
    # do data collection in practice

    import argparse

    parser = argparse.ArgumentParser(description="data_collection")
    parser.add_argument("--ckpt-path", required=True, default=None, type=str)
    parser.add_argument("--env-name", required=True, default="cheetah_run", type=str)
    args = parser.parse_args()

    path = Path(args.ckpt_path)
    with path.open("rb") as f:
        payload = torch.load(f)

    agent = payload["agent"]
    env = make_multiobs(args.env_name, seed=1, frame_stack=3, action_repeat=2)
    n_episodes = 20

    trajs, rews = collect_data(agent, env, n_episodes)
    for k, v in trajs.items():
        print(f"{k}: {len(v), v[0].shape}")

    # print reward stats
    print("reward stats")
    print("============")
    print(np.mean(rews), np.std(rews), np.min(rews), np.max(rews))
    print("============")

    # now save like it is done in generate_v2
    save_path = Path(f"./{args.env_name}_{n_episodes}.pkl")
    with open(save_path, "wb") as fs:
        pickle.dump(trajs, fs)
