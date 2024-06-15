# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

import wandb

warnings.filterwarnings("ignore", category=DeprecationWarning)
import datetime
import os
import tracemalloc

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import suite.dmc as dmc
import utils
from logger import Logger
from replay_buffer import (
    BoostingReplayBufferStorage,
    make_boosting_loader,
    make_expert_replay_loader,
    make_replay_loader,
)
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()
        print("Everything Setup")

        self.agent = make_agent(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            self.cfg.agent,
        )
        print("Agent Init")
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        # Jonathan
        self._best_score = -float("inf")
        self.counter = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.train_env = dmc.make(
            self.cfg.task_name,
            self.cfg.frame_stack,
            self.cfg.action_repeat,
            self.cfg.seed,
        )
        self.eval_env = dmc.make(
            self.cfg.task_name,
            self.cfg.frame_stack,
            self.cfg.action_repeat,
            self.cfg.seed,
        )
        # create replay buffer
        data_specs = (
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )

        self.replay_storage = BoostingReplayBufferStorage(
            data_specs, self.work_dir, self.cfg.n_learners
        )

        self.replay_loader = make_replay_loader(
            self.work_dir,
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
            n_learners=self.cfg.n_learners,
        )
        self.disc_loader = make_boosting_loader(
            self.work_dir,
            self.cfg.num_collect_episodes * 501,  # Make more general... hack for now
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
            n_learners=self.cfg.n_learners,
            eta=self.cfg.eta,
        )
        self._replay_iter = None
        self._disc_replay_iter = None

        expert_dir = Path(self.cfg.expert_dir) / f"{self.cfg.task_name}_10.pkl"
        self.expert_loader = make_expert_replay_loader(
            expert_dir,
            self.cfg.num_demos,
            self.cfg.batch_size,
            self.cfg.nstep,
        )
        self._expert_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @property
    def expert_iter(self):
        if self._expert_iter is None:
            self._expert_iter = iter(self.expert_loader)
        return self._expert_iter

    @property
    def disc_replay_iter(self):
        if self._disc_replay_iter is None:
            self._disc_replay_iter = iter(self.disc_loader)
        return self._disc_replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            # self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(
                        time_step.observation, self.global_step, eval_mode=True
                    )
                time_step = self.eval_env.step(action)
                # self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1
            episode += 1
            # self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

        # Jonathan: Add checkpoint saving
        episode_reward = total_reward / episode
        if episode_reward > self._best_score:
            self.save_checkpoint(int(episode_reward))
            self._best_score = episode_reward
        return episode_reward

    # ===== Boosting Functions =====
    def collect_samples(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_collect_episodes)
        episodes = []
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            time_steps = [time_step]
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(
                        time_step.observation, self.global_step, eval_mode=True
                    )
                time_step = self.eval_env.step(action)
                time_steps.append(time_step)
                total_reward += time_step.reward
                step += 1
            episodes.append(time_steps)
            episode += 1

        weak_learner_avg_reward = total_reward / episode

        if self.cfg.wandb:
            wandb.log({"boosting/weak_learner_returns": weak_learner_avg_reward})
        # Episodes <- List[List[timesteps]]
        return episodes

    def boosting_eval(self):
        print("Boosting EVAL")
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        # weights = self.disc_loader.dataset.get_weights()
        weights = self.disc_loader.dataset._weights
        # TODO: Get weights and then sample per episode

        episodes = []
        while eval_until_episode(episode):
            self.agent.sample_learner(weights)
            time_step = self.eval_env.reset()
            # self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            time_steps = [time_step]
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.boosted_act(
                        time_step.observation, self.global_step, eval_mode=True
                    )
                time_step = self.eval_env.step(action)
                # self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1
                time_steps.append(time_step)
            episodes.append(time_steps)
            episode += 1
            # self.video_recorder.save(f'{self.global_frame}.mp4')

        def ts_to_tensor(ep):
            obs = np.stack([ts.observation for ts in ep[:-1]])
            act = np.stack([ts.action for ts in ep[1:]])
            return dict(observation=obs, action=act)

        divergence_episodes = [ts_to_tensor(ep) for ep in episodes]
        divergence = self.agent.compute_divergence(
            divergence_episodes, self.expert_loader
        )
        if self.cfg.wandb:
            wandb.log({"boosting/divergence": divergence})

        # TODO: add logging capabilities for boosting
        with self.logger.log_and_dump_ctx(self.global_frame, ty="boosting_eval") as log:
            log("episode_reward", total_reward / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

        # Jonathan: Add checkpoint saving
        episode_reward = total_reward / episode
        if episode_reward > self._best_score:
            self.save_checkpoint(int(episode_reward))
            self._best_score = episode_reward
        return episode_reward

    # =============================

    def train(self):
        # predicates
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )
        disc_update_every_step = utils.Every(
            self.cfg.disc_every_frames, self.cfg.action_repeat
        )

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        # self.train_video_recorder.init(time_step.observation)
        metrics = None
        eval_counter = 0
        eval_return = 0
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                # self.train_video_recorder.save(f"{self.global_frame}.mp4")
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward)
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        log("buffer_size", len(self.replay_storage))
                        log("step", self.global_step)
                    if self.cfg.wandb:
                        wandb.log(
                            {
                                "train/fps": episode_frame / elapsed_time,
                                "train/total_time": total_time,
                                "train/episode_reward": episode_reward,
                                "train/episode_length": episode_frame,
                                "train/episode": self.global_episode,
                                "train/global_step": self.global_step,
                                "train/global_frame": self.global_frame,
                            }
                        )
                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                # self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                eval_return = self.eval()
                boosting_eval_return = self.boosting_eval()
                eval_counter += 1

            # ======== DISCRIMINATOR UPDATE ========
            if not seed_until_step(self.global_step) and disc_update_every_step(
                self.global_step
            ):
                # Get Weak Learner
                self.agent.add_learner()

                # Collect Samples From Learner
                episodes = self.collect_samples()
                learner_idx = self.replay_storage.store_batch(episodes)
                self.disc_loader.dataset.reset_size(learner_idx)

                # Update Weights
                self.disc_loader.dataset.get_weights()
                self.disc_loader.dataset._try_fetch(learner_idx)  # fetch data

                # Update Discriminator
                self.agent.update_discriminator(self.disc_replay_iter, self.expert_iter)
                cleared_fns = self.replay_storage.clear_buffer()
                self.replay_loader.dataset._clear_fns(cleared_fns)
                # print('Replay MEM EPS:', len(self.disc_loader.dataset._episodes.keys()))

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(
                    time_step.observation, self.global_step, eval_mode=False
                )

            # ======== POLICY UPDATE ========
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(
                    self.replay_iter, self.expert_iter, self.global_step
                )

                metrics["eval/eval_return"] = eval_return
                metrics["eval/custom_step"] = eval_counter
                metrics["eval/boosting_return"] = boosting_eval_return

                self.logger.log_metrics(metrics, self.global_frame, ty="train")
                if self.cfg.wandb:
                    wandb.log(metrics)
            # ================================

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            # self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        keys_to_save = ["agent", "timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def save_checkpoint(self, score):
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        snapshot = self.work_dir / f"{ts}_{score}_checkpoint.pt"
        keys_to_save = ["agent", "timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        with snapshot.open("rb") as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path="cfgs", config_name="config_boosting")
def main(cfg):
    from train_vision import Workspace as W

    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / "snapshot.pt"
    project_name = cfg.wandb_project_name

    entity = cfg.wandb_entity
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    name = f"{ts}_{cfg.experiment}"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()

    if cfg.wandb:
        group = f"boosting_{cfg.task_name}_{cfg.num_demos}"
        with wandb.init(
            project=project_name, entity=entity, name=name, group=group
        ) as run:
            wandb.define_metric("eval/custom_step")
            wandb.define_metric("eval/*", step_metric="eval/custom_step")
            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step")
            workspace.train()
    else:
        tracemalloc.start()
        workspace.train()


if __name__ == "__main__":
    main()
