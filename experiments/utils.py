from collections import deque
from typing import Any, Dict, List, Tuple, Union

import cv2
import gym
import numpy as np
import torch
from gym import spaces
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, NoopResetEnv
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback

import wandb
from lge.buffer import LGEBuffer
from lge.modules.ae_module import VQVAEModule
from lge.utils import preprocess


def is_atari(env_id: str) -> bool:
    """
    Checks if the given environment id corresponds to an Atari environment.

    Args:
        env_id (str): The id of the environment to check.

    Returns:
        bool: True if the environment is an Atari environment, False otherwise."""
    entry_point = gym.envs.registry.env_specs[env_id].entry_point
    return "AtariEnv" in str(entry_point)


class DoneOnLifeLost(gym.Wrapper):
    """
    A gym wrapper that terminates the environment a life is lost.

    This wrapper is intended to be used with environments that have a "lives" key in their info dictionary,
    which indicates the number of lives the agent has remaining. When the number of lives falls below a certain
    threshold, the environment is terminated and the episode is considered done.

    Args:
        env (gym.Env): Atari environment to wrap.
    """

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = super().step(action)
        lives = info["lives"]
        if lives < 6:
            done = True
            info["dead"] = True
        return obs, reward, done, info


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial state by taking random number of no-ops on reset.

    No-op is assumed to be action 0. Number of no-ops in game frames.

    Args:
        env (gym.Env): Atari environment to wrap
        noop_max (int): Maximum number of no-ops.
    """

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self) -> np.ndarray:
        nb_noops = self.unwrapped.np_random.randint(self.noop_max + 1)
        obs = self.env.reset()
        for _ in range(nb_noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """
    Returns only every `skip`-th frame.

    Repeats the action, sums the reward, and takes the maximum of the last two observations.

    Args:
        env (gym.Env): Atari environment to wrap.
        skip (int): Number of frames to skip. Default: 4.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self.skip = skip

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        # Clear past frame buffer and init. to first obs. from inner env.
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class AtariCeller(gym.Wrapper):
    """
    Add the cell representation the info.

    Args:
        env (gym.Env): Atari environment to wrap.
    """

    def __init__(self, env: gym.Env, width: int = 11, height: int = 8, nb_pixels: int = 8) -> None:
        super().__init__(env)
        self.width, self.height, self.nb_pixels = width, height, nb_pixels

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = super().step(action)
        cell = cv2.resize(obs, (self.width, self.height))
        cell = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
        cell = (np.floor(cell / 255 * self.nb_pixels)).astype(np.uint8)
        info["cell"] = cell
        return obs, reward, done, info


class GrayscaleDownscale(gym.ObservationWrapper):
    """
    Downscale and grayscale the observation.

    Args:
        env (gym.Env): Atari environment to wrap.
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84) -> None:
        super().__init__(env)
        self.width, self.height = width, height
        self.observation_space = spaces.Box(0, 255, shape=(self.width, self.height, 1), dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        observation = cv2.resize(observation, (self.width, self.height))
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = np.expand_dims(observation, 2)  # (W, H) to (W, H, 1)
        return observation


class AtariWrapper(gym.Wrapper):
    """
    Wrapper for Atari environments.

    This wrapper applies several Atari-specific modifications to the environment. These modifications include:

    - Terminate the environment when a life is lost.
    - Takes random number of no-ops on reset.
    - Action sticking: The agent take the previous action with a given probability.
    - Frame skiping and max pooling.
    - Add to info the cell representation
    - Grayscale and downscale observation to 84 x 84.

    Args:
        env (gym.Env): Atari environment to wrap.
    """

    def __init__(self, env: gym.Env) -> None:
        env = DoneOnLifeLost(env)  # In game frame, only affect the step
        env = NoopResetEnv(env)  # In game frame, only affect the reset
        env = MaxAndSkipEnv(env)
        env = AtariCeller(env)
        env = GrayscaleDownscale(env)
        super().__init__(env)


class MaxInnerRewardLogger(BaseCallback):
    """
    Callback that logs the maximum reward observed in the replay buffer at a given frequency.

    Args:
        freq (int): Frequency at which the maximum reward is logged. Default is 1_000
        verbose (int): Verbosity level, default is 0
    """

    def __init__(self, freq: int = 1_000, verbose: int = 0):
        super().__init__(verbose)
        self.freq = freq
        self.max_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            buffer = self.locals["replay_buffer"]  # ReplayBuffer
            infos = buffer.infos
            if not buffer.full:
                if buffer.pos == 0:
                    return True
                infos = infos[: buffer.pos]
            rewards = np.array([info.get("env_reward") for info in infos.flatten()])
            self.max_reward = max(np.max(rewards), self.max_reward)
            self.logger.record("env/max_reward", self.max_reward)
        return True


class MaxRewardLogger(BaseCallback):
    """
    Callback that logs the maximum reward observed in the replay buffer at a given frequency.

    Args:
        freq (int): Frequency at which the maximum reward is logged. Default is 1_000
        verbose (int): Verbosity level, default is 0
    """

    def __init__(self, freq: int = 1_000, verbose: int = 0):
        super().__init__(verbose)
        self.freq = freq
        self.max_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            buffer = self.locals["replay_buffer"]  # ReplayBuffer
            rewards = buffer.rewards
            if not buffer.full:
                if buffer.pos == 0:
                    return True
                rewards = rewards[: buffer.pos]
            self.max_reward = max(np.max(rewards), self.max_reward)
            self.logger.record("env/max_reward", self.max_reward)
        return True

class AtariNumberCellsLogger(BaseCallback):
    """
    Callback to monitor and log the number of unique cells visited in Atari environments.

    It does so by monitoring the replay buffer and retrieving the cell information for each
    transition stored in the buffer, with a logging frequency specified by the user.

    Args:
        freq (int): Frequency at which the number of unique cells visited is logged. Default is 1_000
        verbose (int): Verbosity level, default is 0
    """

    def __init__(self, freq: int = 1_000, verbose: int = 0):
        super().__init__(verbose)
        self.all_cells = np.zeros((0, 8, 11), dtype=np.uint8)
        self.freq = freq
        self._last_call = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            buffer = self.locals["replay_buffer"]  # type: LGEBuffer
            infos = buffer.infos
            if buffer.pos < self._last_call:
                idxs = np.arange(self._last_call, buffer.pos + buffer.buffer_size) % buffer.buffer_size
            else:
                idxs = np.arange(self._last_call, buffer.pos)
            infos = infos[idxs]
            cells = np.array([info[env_idx]["cell"] for info in infos for env_idx in range(buffer.n_envs)])
            self.all_cells = np.concatenate((self.all_cells, cells))
            self.all_cells = np.unique(self.all_cells, axis=0)
            self.logger.record("env/nb_cells", len(self.all_cells))
            self._last_call = buffer.pos
        return True


class NumberCellsLogger(BaseCallback):
    def __init__(self, freq: int = 1_000, verbose: int = 0):
        super().__init__(verbose)
        self.freq = freq
        self._last_call = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            buffer = self.locals["replay_buffer"]  # type: LGEBuffer
            observations = buffer.next_observations["observation"]
            if buffer.pos < self._last_call:
                idxs = np.arange(self._last_call, buffer.pos + buffer.buffer_size) % buffer.buffer_size
            else:
                idxs = np.arange(self._last_call, buffer.pos)
            observations = observations[idxs]
            observations = np.reshape(observations, (-1, observations.shape[-1]))  # (N, N_ENVS, D) to (N*N_ENVS, D)
            cells = np.floor(observations)
            if hasattr(self, "all_cells"):
                self.all_cells = np.concatenate((self.all_cells, cells))
            else:
                self.all_cells = cells
            self.all_cells = np.unique(self.all_cells, axis=0)
            self.logger.record("env/nb_cells", len(self.all_cells))
            self._last_call = buffer.pos
        return True


class DensityLogger(BaseCallback):
    """
    Callback that logs the latent density and push it as an histogram on Wandb.

    Args:
        freq (int): Frequency at which the density is logged. Default is 1_000
        verbose (int): Verbosity level, default is 0
    """

    def __init__(self, freq: int = 1_000, verbose: int = 0):
        super().__init__(verbose)
        self.freq = freq

    def _on_step(self):
        """Callback that logs the density of the replay buffer at a given frequency."""
        if self.n_calls % self.freq == 0:
            buffer = self.model.replay_buffer  # LGEBuffer
            density = buffer.density.copy()
            wandb.log({"densities": wandb.Histogram(density)})
        return True


class TopGoalsLogger(BaseCallback):
    """
    Log the top 25 goals to Wandb, i.e. the observations that have the lowest latent density.

    Args:
        freq (int): Frequency at which the top goals are logged. Default is 1_000
        verbose (int): Verbosity level of the callback. Default is 0
    """

    def __init__(self, freq: int = 1_000, verbose: int = 0):
        super().__init__(verbose)
        self.freq = freq

    def _on_step(self):
        if self.n_calls % self.freq == 0:
            buffer = self.model.replay_buffer
            if len(buffer.sorted_density) > 0:
                top_goals = buffer.sorted_density[:25]
                top_goal_pos, top_goals_env = np.unravel_index(top_goals, (buffer.buffer_size, buffer.n_envs))
                top_obs = buffer.next_observations["observation"][top_goal_pos, top_goals_env]
                images = np.vstack(np.dstack(np.reshape(np.transpose(top_obs, (0, 2, 3, 1)), (5, 5, 84, 84, 1))))
                images = wandb.Image(images, caption="Top goals")
                wandb.log({"Top goals": images})
        return True


class GoalTrajectoriesLogger(BaseCallback):
    """
    Log the goal trajectories to Wandb.
    A goal trajectory is the sequence of observations that lead to the goal being achieved.

    Args:
        freq (int): Frequency at which the goal trajectories are logged. Default is 1_000
        verbose (int): Verbosity level of the callback. Default is 0
    """

    def __init__(self, freq: int = 1_000, verbose: int = 0):
        super().__init__(verbose)
        self.freq = freq

    def _on_step(self):
        if self.n_calls % self.freq == 0:
            goal_trajectories = self.training_env.goal_trajectories
            image_array = [np.hstack(traj) for traj in goal_trajectories]
            max_width = max([image.shape[1] for image in image_array])
            image_array = [
                np.pad(image, [(0, 0), (0, max_width - image.shape[1]), (0, 0)], mode="constant", constant_values=0)
                for image in image_array
            ]
            image = wandb.Image(np.vstack(image_array), caption="Goals trajectories")
            wandb.log({"Goal trajectories": image})
        return True


class VQVAELogger(BaseCallback):
    """
    Log reconstructions of observations from the VQVAE model to Wandb.

    Args:
        module (VQVAEModule): The VQVAE module to be used for generating reconstructions
        freq (int): Frequency at which reconstructions are logged. Default is 1_000
        verbose (int): Verbosity level of the callback. Default is 0
    """

    def __init__(self, module: VQVAEModule, freq: int = 1_000, verbose: int = 0):
        super().__init__(verbose)
        self.module = module
        self.freq = freq

    def _on_step(self):
        if self.n_calls % self.freq == 0:
            buffer = self.model.replay_buffer  # LGEBuffer
            samples = buffer.sample(8)
            observations = samples.next_observations
            observations = preprocess(observations, buffer.observation_space)["observation"]
            recons, _ = self.module(observations)
            observations, recons = torch.cat(tuple(observations), dim=2), torch.cat(tuple(recons), dim=2)
            image_array = torch.cat((observations, recons), dim=1)
            wandb.log({"VQVAE": wandb.Image(image_array, caption="VQVAE")})

        return True


class ReplayBufferWithInfo(ReplayBuffer):
    """
    A replay buffer that also stores additional information provided by the environment.
    The information is stored as an array of dictionaries, where the dictionary keys depend on the environment.

    Args:
        buffer_size (int): Maximum number of transitions that can be stored in the buffer.
        observation_space (spaces.Space): The observation space of the environment.
        action_space (spaces.Space): The action space of the environment.
        device (Union[torch.device, str]): Device.
        n_envs (int): Number of parallel environments.
        optimize_memory_usage (bool): If true, the buffer will optimize memory usage by storing the data in uint8
            format, but this will make indexing slower.
        handle_timeout_termination (bool): Whether to handle the case when an episode termination is caused by the
            environment signaling a timeout (e.g. in `gym.Env`) by storing the episode in the buffer and treating the
            next observation as the start of a new episode.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination
        )
        self.infos = np.zeros((self.buffer_size, self.n_envs), dtype=object)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        pos = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        self.infos[pos] = infos
