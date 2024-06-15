import datetime
import functools
import io
import pickle as pkl
import random
import traceback
from collections import defaultdict, deque

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

import utils


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def expert_len(episode):
    return next(iter(episode.values())).shape[0]


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


@functools.cache
def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


# ============================ DISK STORAGE OBJECTS ============================


class ReplayBufferStorage:
    """Disk storage object for vision-based control tasks.

    Organized as follows:

    work_dir
    - buffer
    - - episodes were saved

    Boosting 2 learners
    work_dir
    - buffer
    - - temporary episodes for RL
    - 0_buffer
    - - episodes from learner_0
    - 1_buffer
    - - episodes from learner_1
    """

    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec in self._data_specs:
            if spec.name == "next_observation":
                continue
            else:
                value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)

        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob("*.npz"):
            _, _, eps_len = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{eps_idx}_{eps_len}.npz"
        save_episode(episode, self._replay_dir / eps_fn)


class BoostingReplayBufferStorage:
    """
    DAC / DRQ-v2
    work_dir
    - buffer
    - - episodes were saved

    Boosting 2 learners
    work_dir
    - buffer
    - - temporary episodes for RL
    - 0_buffer
    - - episodes from learner_0
    - 1_buffer
    - - episodes from learner_1
    """

    def __init__(self, data_specs, work_dir, n_learners):
        replay_dir = work_dir / "buffer"
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()
        self.n_learners = n_learners
        self.current_learner_idx = 0
        self.work_dir = work_dir

        for idx in range(n_learners):
            tmp_dir = work_dir / f"{idx}_buffer"
            tmp_dir.mkdir(exist_ok=True)

    def __len__(self):
        return self._num_transitions

    def add(self, time_step, save_dir=None):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode, save_dir=save_dir)

    def clear_buffer(self):
        # To use to Clear out Replay buffer samples not needed
        # for discriminator update
        fns = self._replay_dir.glob("*.npz")
        for fn in fns:
            fn.unlink(missing_ok=True)
        return fns

    def store_batch(self, episodes):
        save_dir = self.work_dir / f"{self.current_learner_idx}_buffer"

        # Delete previous
        for fn in save_dir.glob("*.npz"):
            fn.unlink(missing_ok=True)

        # Populate
        for ep in episodes:
            for ts in ep:
                self.add(ts, save_dir=save_dir)
        prev_idx = self.current_learner_idx
        # Update Learner Pointer
        self.current_learner_idx = (self.current_learner_idx + 1) % self.n_learners
        return prev_idx

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob("*.npz"):
            _, _, eps_len = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode, save_dir=None):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{eps_idx}_{eps_len}.npz"
        if save_dir is None:
            save_episode(episode, self._replay_dir / eps_fn)
        else:
            save_episode(episode, save_dir / eps_fn)


# ============================ PYTORCH DATASET OBJECTS/ITERABLES ============================


class ReplayBuffer(IterableDataset):
    """Generic DrQv2 replay buffer."""

    def __init__(
        self,
        replay_dir,
        max_size,
        num_workers,
        nstep,
        discount,
        fetch_every,
        save_snapshot,
        return_one_step,
    ):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._return_one_step = return_one_step

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx + self._nstep - 1]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])

        # 1 step next obs
        for i in range(self._nstep):
            step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            discount *= episode["discount"][idx + i] * self._discount
        if self._return_one_step:
            # This is for r(s, s') for imitation
            one_step_obs = episode["observation"][idx]
            return (obs, action, reward, discount, next_obs, one_step_obs)
        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


class BoostingReplayBuffer(IterableDataset):
    """Boosting replay buffer for vision-based tasks."""

    def __init__(
        self,
        work_dir,
        max_size,
        num_workers,
        nstep,
        discount,
        fetch_every,
        save_snapshot,
        n_learners,
        eta,
    ):
        print("Boosting", max_size)
        # Boosting Parameters
        self._n_learners = n_learners
        self._eta = eta
        self._weights = None
        self._counter = 0

        # For each learner
        self._replay_dirs = [work_dir / f"{idx}_buffer" for idx in range(n_learners)]

        self._size = [0 for _ in range(n_learners)]
        self._max_size = max_size  # Max Size per Learner
        self._num_workers = max(1, num_workers)
        self._episode_fns = [[] for _ in range(n_learners)]
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

    def get_learner_weights(self):
        # For evaluation
        if self._weights is None:
            return None

        # In accending order
        return np.sort(self._weights)

    def reset_size(self, idx):
        self._size[idx] = 0
        for eps_fn in self._episode_fns[idx]:
            if eps_fn in self._episodes.keys():
                del self._episodes[eps_fn]
        self._episode_fns[idx] = []

    def reset_eps(self):
        self._episodes = dict()

    def get_weights(self):
        # Uniform Sampling
        self._counter += 1
        if self._counter < 2:
            self._weights = None
            return

        # This is the case when n_learners = 2
        if self._weights is None:
            self._weights = np.array([(1 - self._eta), self._eta])
            return

        # Polyak Averaging for every weak_learner added
        self._weights *= 1 - self._eta
        if self._counter <= self._n_learners:
            self._weights = np.append(self._weights, [self._eta])
            return

        idx = (self._counter - 1) % self._n_learners

        self._weights[idx] = self._eta
        if idx == (self._n_learners - 1):
            self._weights[0] /= self._eta
        else:
            self._weights[idx + 1] /= self._eta

    def _sample_episode(self):
        # Sample Learner
        learner_idx = np.random.choice(np.arange(self._counter), p=self._weights)
        # Sample Episode from learner
        eps_fn = random.choice(self._episode_fns[learner_idx])
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn, learner_idx):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        print(f"online episode length: {eps_len}")
        # while eps_len + self._size[learner_idx] > self._max_size:
        #    early_eps_fn = self._episode_fns[learner_idx].pop(0)
        #    early_eps = self._episodes.pop(early_eps_fn)
        #    self._size[learner_idx] -= episode_len(early_eps)
        self._episode_fns[learner_idx].append(eps_fn)
        self._episode_fns[learner_idx].sort()
        self._episodes[eps_fn] = episode
        self._size[learner_idx] += eps_len

        return True

    def _try_fetch(self, learner_idx):
        # if self._samples_since_last_fetch < self._fetch_every:
        #    return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        learner_dir = self._replay_dirs[learner_idx]
        eps_fns = sorted(learner_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            # if fetched_size + eps_len > self._max_size:
            #    break
            fetched_size += eps_len
            if not self._store_episode(eps_fn, learner_idx):
                break

    def _sample(self):
        # try:
        #    self._try_fetch()
        # except:
        #    traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx + self._nstep - 1]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        for i in range(self._nstep):
            step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            discount *= episode["discount"][idx + i] * self._discount

        # To compute Nstep Online
        int_obs = episode["observation"][idx : idx + self._nstep - 1]
        int_act = episode["action"][idx + 1 : idx + self._nstep]
        return (obs, action, reward, discount, next_obs, int_obs, int_act)

    def __iter__(self):
        while True:
            yield self._sample()


class ReplayBufferLocal(IterableDataset):
    """DrQv2 replay buffer (using disk storage) with boosting functionality added in."""

    def __init__(
        self,
        storage,
        max_size,
        num_workers,
        nstep,
        discount,
        fetch_every,
        save_snapshot,
        return_one_step,
        eta=None,
        n_episodes=None,
    ):
        self._storage = storage
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._return_one_step = return_one_step

        # Boosting Specific
        self._idx = 0
        self._eta = eta
        self._n_episodes = n_episodes

        if self._eta is not None:
            assert self._n_episodes is not None

            self._n_samples = (
                self._n_episodes * 1000
            )  # length of 1 episode is 1000 for DMC, TODO delete if we don't need it
            self._weights = None

    def get_learner_weights(self):
        # For evaluation

        if self._weights is None:
            return None

        policy_weights = np.unique(
            self._weights * self._n_episodes
        )  # this is because episodes, not timesteps are always stored on disk in local replay buffer
        policy_weights.sort()

        # In accending order
        return policy_weights

    def get_weights(self):
        n_learners = len(self) // self._n_episodes

        # Uniform Sampling
        if n_learners < 2:
            self._weights = None
            return

        # This is the case when n_learners = 2
        if self._weights is None:
            uniform_weights = np.full(
                (len(self)), 1 / self._n_episodes, dtype=np.float32
            )
            uniform_weights[: self._n_episodes] *= 1 - self._eta
            uniform_weights[self._n_episodes : self._idx] *= self._eta
            self._weights = uniform_weights
            return

        # Polyak Averaging for every weak_learner added
        self._weights *= 1 - self._eta
        new_weights = np.full(
            (self._n_episodes), self._eta / self._n_episodes, dtype=np.float32
        )
        if not self._full:
            self._weights = np.concatenate([self._weights, new_weights])
            return

        self._weights[self._idx - self._n_episodes : self._idx] = new_weights
        self._weights[self._idx : self._idx + self._n_episodes] /= self._eta

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._storage._replay_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx + self._nstep - 1]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        for i in range(self._nstep):
            step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            discount *= episode["discount"][idx + i] * self._discount
        if self._return_one_step:
            one_step_obs = episode["observation"][idx]
            return (obs, action, reward, discount, next_obs, one_step_obs)
        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


class ReplayBufferMemory:
    """In-memory replay buffer, with boosting weights."""

    def __init__(
        self,
        specs,
        max_size,
        batch_size,
        nstep,
        discount,
        eta=None,
        n_samples=None,
        roll_for_overflow_change=False,
    ):
        print(f"specs={specs}")
        self._specs = specs
        self._max_size = max_size
        self._batch_size = batch_size
        self._nstep = nstep
        self._discount = discount
        self._roll_for_overflow_change = roll_for_overflow_change
        self._idx = 0
        self._full = False
        self._items = dict()
        self._queue = deque([], maxlen=nstep + 1)
        for spec in specs:
            self._items[spec.name] = np.empty((max_size, *spec.shape), dtype=spec.dtype)

        # Save the intermediates for on-policy calculation....
        if nstep > 1:
            for i in range(1, nstep):
                self._items[f"obs_{i}"] = np.empty(
                    (max_size, *specs[0].shape),
                    dtype=specs[0].dtype,
                )
                self._items[f"act_{i}"] = np.empty(
                    (max_size, *specs[1].shape), dtype=specs[1].dtype
                )

        self._eta = eta
        self._n_samples = n_samples

        if self._eta is not None:
            assert self._n_samples is not None

            # in the case of sampling, we must decrease the number of samples so as to weight appropriately
            if nstep > 1:
                self._n_samples = (
                    self._n_samples - nstep - 1
                )  # this is the number of samples that will be collected by Workspace.collect_samples() when populating the discriminator buffer

            self._weights = None

        self._precision = np.float32 if max_size <= 1000000 else np.float64
        print(self._precision)

    def __len__(self):
        return self._max_size if self._full else self._idx

    def get_learner_weights(self):
        # For evaluation

        if self._weights is None:
            return None

        weights = []
        for idx in range(0, len(self), self._n_samples):
            weights.append(self._weights[idx] * self._n_samples)
        print(weights)

        policy_weights = np.unique(
            self._weights * self._n_samples
        )  # why should this have exactly n_learners values all the time -- cuz max_size % n_samples = 0 right
        print(f"current learner weights: {policy_weights}")
        policy_weights.sort()  # sort bc most recent policy gets higher weights

        # In ascending order
        return policy_weights

    def get_weights(self):
        n_learners = len(self) // self._n_samples

        # Uniform Sampling
        if n_learners < 2:
            self._weights = None
            return

        # This is the case when n_learners = 2, we will only have collected `2 * n_samples` samples by this point, so the distribution is fine...
        if self._weights is None:
            uniform_weights = np.full(
                (len(self)), 1 / self._n_samples, dtype=self._precision
            )
            uniform_weights[: self._n_samples] *= 1 - self._eta
            uniform_weights[self._n_samples : self._idx] *= self._eta
            self._weights = uniform_weights
            return

        # Polyak Averaging for every weak_learner added (add n_samples from the latest weak learner, and reweight)
        new_weights = np.full(
            (self._n_samples), self._eta / self._n_samples, dtype=self._precision
        )
        if not self._full:
            self._weights *= 1 - self._eta
            self._weights = np.concatenate([self._weights, new_weights])
            # print(f"new weights after setting / sum (when not full): {self._weights} / {np.sum(self._weights)}")
            return

        # print('=== reached nondebugged territory ===')
        # print(f'current index: {self._idx}')
        # print(f'current number of samples: {self._n_samples}')
        # print(f'current weights shape: {new_weights.shape}')
        print(f"current index: {self._idx}")
        print(f"current size: {len(self)}")
        print(f"weights size: {self._weights.shape}")

        # in case there is a mismatch
        if self._weights.shape[0] < self._max_size:
            # need to add new weights in to reach max size (we know the sum of the weights is 1, so multiply by 1 - eta and add in)
            self._weights *= 1 - self._eta
            self._weights = np.concatenate([self._weights, new_weights])
        else:
            # don't change weights shape (already at max size)
            if not self._roll_for_overflow_change:
                # pre update sum (these are the weights for the oldest learner still in the buffer, so the ones we're going to replace)
                if self._idx == 0:
                    curr_weight_sum = np.sum(self._weights[-self._n_samples :])
                    self._weights[-self._n_samples :] = new_weights
                else:
                    curr_weight_sum = np.sum(
                        self._weights[self._idx - self._n_samples : self._idx]
                    )
                    self._weights[self._idx - self._n_samples : self._idx] = new_weights

                # scale all other weights by renormalizing them to sum to 1
                ratio = (1 - self._eta) / (1 - curr_weight_sum)
                if self._idx == 0:
                    self._weights[: -self._n_samples] *= ratio
                else:
                    self._weights[: self._idx - self._n_samples] *= ratio
                    self._weights[self._idx :] *= ratio
            else:
                self._weights = np.roll(self._weights, shift=self._n_samples)

        # print(f"new weights after setting / sum (when full): {self._weights} / {np.sum(self._weights)}")
        print(f"weights sum: {self._weights.sum()}")

    def get_buffer(self):
        return self._items, self._full, self._idx

    def load_buffer(self, items, full, idx):
        self._items = items
        self._full = full
        self._idx = idx
        self._queue = deque([], maxlen=self._nstep + 1)

    def add(self, time_step):
        for spec in self._specs:
            if spec.name == "next_observation":
                continue
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            # assert spec.shape == value.shape and spec.dtype == value.dtype

        # add to observation queue
        self._queue.append(time_step)

        # if we have enough samples, we add to the buffer items (index only updates here -- if nstep > 1 what happens?)
        if len(self._queue) == self._nstep + 1:
            np.copyto(self._items["observation"][self._idx], self._queue[0].observation)
            np.copyto(self._items["action"][self._idx], self._queue[1].action)
            np.copyto(
                self._items["next_observation"][self._idx], self._queue[-1].observation
            )

            if self._nstep > 1:
                for i in range(1, self._nstep):
                    np.copyto(
                        self._items[f"obs_{i}"][self._idx], self._queue[i].observation
                    )
                    np.copyto(
                        self._items[f"act_{i}"][self._idx], self._queue[i + 1].action
                    )

            reward, discount = 0.0, 1.0
            self._queue.popleft()
            for ts in self._queue:
                reward += discount * ts.reward
                discount *= ts.discount * self._discount
            np.copyto(self._items["reward"][self._idx], reward)
            np.copyto(self._items["discount"][self._idx], discount)

            self._idx = (self._idx + 1) % self._max_size
            self._full = self._full or self._idx == 0

        if time_step.last():
            self._queue.clear()

    def _sample(self):

        if not self._eta:
            idxs = np.random.randint(0, len(self), size=self._batch_size)
        else:
            idxs = np.random.choice(
                np.arange(len(self)), size=self._batch_size, p=self._weights
            )
        batch = tuple(self._items[spec.name][idxs] for spec in self._specs)

        # NOTE: Not eta since this is just for Policy
        if not self._eta and self._nstep > 1:
            int_obs = tuple(
                self._items[f"obs_{i}"][idxs] for i in range(1, self._nstep)
            )
            int_act = tuple(
                self._items[f"act_{i}"][idxs] for i in range(1, self._nstep)
            )
            batch = batch + int_obs + int_act
        return batch

    def __iter__(self):
        while True:
            yield self._sample()


class ReplayWrapper:
    def __init__(
        self,
        train_env,
        data_specs,
        work_dir,
        cfg,
        buffer_name="buffer",
        return_one_step=False,
    ):
        self.cfg = cfg
        if cfg.buffer_local:
            # meaning we have high-dimensional data, and therefore it is very hard to store all in RAM
            self.replay_storage = ReplayBufferStorage(
                data_specs, work_dir / buffer_name
            )

            max_size_per_worker = cfg.replay_buffer_size // max(
                1, cfg.replay_buffer_num_workers
            )

            iterable = ReplayBufferLocal(
                self.replay_storage,
                max_size_per_worker,
                cfg.replay_buffer_num_workers,
                cfg.nstep,
                cfg.suite.discount,
                fetch_every=1000,
                save_snapshot=True,
                return_one_step=return_one_step,
            )

            self.replay_buffer = DataLoader(
                iterable,
                batch_size=cfg.batch_size,
                num_workers=cfg.replay_buffer_num_workers,
                pin_memory=True,
                worker_init_fn=_worker_init_fn,
            )
        else:
            # this is where we can just do state-based training, so can store all data in RAM
            self.replay_buffer = ReplayBufferMemory(
                specs=train_env.specs(),
                max_size=cfg.replay_buffer_size,
                batch_size=cfg.batch_size,
                nstep=cfg.nstep,
                discount=cfg.suite.discount,
            )

    def add(self, elt):
        if self.cfg.buffer_local:
            self.replay_storage.add(elt)
        else:
            self.replay_buffer.add(elt)


class ExpertReplayBuffer(IterableDataset):
    """
    Expert replay buffer used for Adversarial IL type algorithms.
    """

    def __init__(self, dataset_path, num_demos, n_step, obs_key="states"):
        # Load Expert Demos
        with open(dataset_path, "rb") as f:
            data = pkl.load(f)
            # obs, act = data[0], data[-2]
            if isinstance(data, list):
                data = utils.list2dict(data, n_trajs=num_demos)

            obs, act = np.array(data[obs_key]), np.array(data["actions"])
            print(obs.shape, act.shape)

        self.obs = obs
        self.act = act
        self.n_step = n_step

        self._episodes = []
        for i in range(num_demos):
            episode = dict(observation=obs[i], action=act[i])
            self._episodes.append(episode)
        self.num_demos = num_demos

    def _sample_episode(self):
        idx = np.random.randint(0, self.num_demos)
        return self._episodes[idx]

    def _sample(self):
        episode = self._sample_episode()
        idx = np.random.randint(0, expert_len(episode) - self.n_step)
        obs = episode["observation"][idx]
        action = episode["action"][idx]
        # if len(action.shape) == 3:
        #     action = np.squeeze(action, axis=1)
        next_obs = episode["observation"][idx + self.n_step]
        if self.n_step > 1:
            int_obs = []
            int_act = []
            for i in range(1, self.n_step):
                int_obs.append(episode["observation"][idx + i])
                int_act.append(episode["action"][idx + i])
            return (obs, action, next_obs) + tuple(int_obs) + tuple(int_act)
        return (obs, action, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


def make_expert_replay_loader(
    dataset_path, num_demos, batch_size, n_step, obs_key, n_workers=2
):
    iterable = ExpertReplayBuffer(dataset_path, num_demos, n_step, obs_key)
    loader = DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )

    return loader


def make_replay_loader(
    replay_dir,
    max_size,
    batch_size,
    num_workers,
    save_snapshot,
    nstep,
    discount,
    return_one_step,
):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(
        replay_dir,
        max_size_per_worker,
        num_workers,
        nstep,
        discount,
        fetch_every=1000,
        save_snapshot=save_snapshot,
        return_one_step=return_one_step,
    )

    loader = DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader


def make_boosting_loader(
    replay_dir,
    max_size,
    batch_size,
    save_snapshot,
    nstep,
    discount,
    n_learners,
    eta,
):
    # NOTE: Max size is size for EACH learner
    max_size_per_worker = max_size // max(1, 1)

    iterable = BoostingReplayBuffer(
        replay_dir,
        max_size_per_worker,
        1,
        nstep,
        discount,
        fetch_every=1000,
        save_snapshot=save_snapshot,
        n_learners=n_learners,
        eta=eta,
    )

    loader = DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader
