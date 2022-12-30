import cv2
import gym
import numpy as np
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv, WarpFrame
from stable_baselines3.common.callbacks import BaseCallback

import wandb
from lge.buffer import LGEBuffer


class AtariWrapper(gym.Wrapper):
    """
    Atari 2600 preprocessings

    Specifically:

    * NoopReset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost.
    * Resize to a square image: 84x84 by default
    * Grayscale observation

    :param env: Atari environment
    :param frame_skip: Frequency at which the agent experiences the game.
    :param screen_size: Resize Atari frame
    """

    def __init__(self, env: gym.Env, frame_skip: int = 4, screen_size: int = 84):
        env = MaxAndSkipEnv(env, skip=frame_skip)
        env = EpisodicLifeEnv(env)
        env = FireResetEnv(env)
        env = WarpFrame(env, width=screen_size, height=screen_size)
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info["ram"] = self.unwrapped.ale.getRAM()
        return obs, reward, done, info


class MaxRewardLogger(BaseCallback):
    def __init__(self, freq: int = 5_000, verbose: int = 0):
        super().__init__(verbose)
        self.freq = freq
        self.max_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            buffer = self.locals["replay_buffer"]  # type: LGEBuffer
            infos = buffer.infos
            if not buffer.full:
                if buffer.pos == 0:
                    return True
                infos = buffer.infos[: buffer.pos]
            rewards = [
                info[env_idx]["episode"]["i"]
                for info in infos
                for env_idx in range(buffer.n_envs)
                if "episode" in info[env_idx]
            ]
            self.max_reward = max(np.max(rewards), self.max_reward)
            self.logger.record("env/max_env_reward", self.max_reward)
        return True


class AtariNumberCellsLogger(BaseCallback):
    def __init__(self, freq: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.all_cells = np.zeros((0, 10, 10), dtype=np.uint8)
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
            observations = np.reshape(observations, (-1, 1, 84, 84))  # (N, N_ENVS, C, H, W) to (N*N_ENVS, C, H, W)
            observations = np.moveaxis(observations, 1, -1)  # (N*N_ENVS, C, H, W) to (N*N_ENVS, H, W, C)
            cells = np.zeros((observations.shape[0], 10, 10), np.uint8)
            for i, observation in enumerate(observations):
                cell = cv2.resize(observation, (10, 10))  #  (N*N_ENVS, H, W, C) to (N*N_ENVS, 20, 20)
                cells[i] = np.floor(cell / 255 * 6) * 255 / 6  # [0, d]
            self.all_cells = np.concatenate((self.all_cells, cells))
            self.all_cells = np.unique(self.all_cells, axis=0)
            self.logger.record("env/nb_cells", len(self.all_cells))
            self._last_call = buffer.pos
        return True


class NumberCellsLogger(BaseCallback):
    def __init__(self, freq: int = 500, verbose: int = 0):
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


def is_atari(env_id: str) -> bool:
    entry_point = gym.envs.registry.env_specs[env_id].entry_point
    return "AtariEnv" in str(entry_point)


class GoalLogger(BaseCallback):
    def __init__(self, freq: int = 1_000, verbose: int = 0):
        super().__init__(verbose)
        self.freq = freq

    def _on_step(self):
        if self.n_calls % self.freq == 0:
            goal_trajectory = self.training_env.get_attr("goal_trajectory")
            image_array = [np.hstack(traj) for traj in goal_trajectory]
            max_width = max([image.shape[1] for image in image_array])
            image_array = [
                np.pad(image, [(0, 0), (0, max_width - image.shape[1]), (0, 0)], mode="constant", constant_values=0)
                for image in image_array
            ]
            images = np.vstack(image_array)

            images = wandb.Image(images, caption="Goals trajectories")
            wandb.log({"Goal trajectory": images}, step=self.num_timesteps)
        return True
