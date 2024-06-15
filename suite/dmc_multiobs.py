# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import OrderedDict, deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs

MANIP_PIXELS_KEY = "pixels"


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    random_feat: Any = None

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


def robust_len(x):
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return 1
        return len(x)
    else:
        return 1


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key="pixels"):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = OrderedDict()
        # print("physics shape", self._env.physics.get_state().shape)
        self.states = np.array([])
        physics = self._env._task.get_observation(self._env._physics)
        for key in physics.keys():
            self.states = np.append(self.states, physics[key])

        count = sum(
            robust_len(i)
            for i in self._env._task.get_observation(self._env._physics).values()
        )
        self._obs_spec["state"] = specs.Array(
            shape=(count,), dtype=np.float32, name="state"
        )

        dim = 0
        for key in wrapped_obs_spec.keys():
            if key != MANIP_PIXELS_KEY:
                spec = wrapped_obs_spec[key]
                assert spec.dtype == np.float64
                assert type(spec) == specs.Array
                dim += np.prod(spec.shape)

        self._obs_spec["features"] = specs.Array(
            shape=(dim,), dtype=np.float32, name="features"
        )

        pixels_spec = wrapped_obs_spec["pixels"]

        self._obs_spec["pixels"] = specs.BoundedArray(
            shape=np.concatenate(
                [[pixels_spec.shape[2] * num_frames], pixels_spec.shape[:2]], axis=0
            ),
            dtype=pixels_spec.dtype,
            minimum=0,
            maximum=255,
            name="observation",
        )

    def append_states(self):
        states = np.array([])
        physics = self._env._task.get_observation(self._env._physics)
        for key in physics.keys():
            states = np.append(states, physics[key])
        return states

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        # obs = np.concatenate(list(self._frames), axis=0)
        # return time_step._replace(observation=obs)
        obs = OrderedDict()
        # obs['state'] = self._env.physics.get_state().copy()
        obs["state"] = self.append_states()
        obs["pixels"] = np.concatenate(list(self._frames), axis=0)
        features = []
        for key, value in time_step.observation.items():
            if key != MANIP_PIXELS_KEY:
                features.append(value.ravel())
        obs["features"] = np.concatenate(features, axis=0).astype(np.float32)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def specs(self):
        obs_spec = self._env.observation_spec()["pixels"]
        action_spec = self._env.action_spec()
        next_obs_spec = specs.Array(obs_spec.shape, obs_spec.dtype, "next_observation")
        reward_spec = specs.Array((1,), action_spec.dtype, "reward")
        discount_spec = specs.Array((1,), action_spec.dtype, "discount")
        return (obs_spec, action_spec, reward_spec, discount_spec, next_obs_spec)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(name, frame_stack, action_repeat, seed):
    domain, task = name.split("_", 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup="ball_in_cup").get(domain, domain)
    visualize_reward = False
    # make sure reward is not visualized

    assert (
        domain,
        task,
    ) in suite.ALL_TASKS, "Not implemented environment creation for non-DMC tasks."

    env = suite.load(domain, task, task_kwargs={"random": seed}, visualize_reward=False)

    pixels_key = "pixels"

    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)

    # zoom in camera for quadruped
    camera_id = dict(quadruped=2).get(domain, 0)
    render_kwargs = dict(height=84, width=84, camera_id=camera_id)
    env = pixels.Wrapper(env, pixels_only=False, render_kwargs=render_kwargs)
    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = ExtendedTimeStepWrapper(env)

    return env
