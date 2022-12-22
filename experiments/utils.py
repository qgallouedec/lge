import cv2
import gym
import numpy as np
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv, WarpFrame
from stable_baselines3.common.callbacks import BaseCallback


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
    * Clip reward to {-1, 0, 1}

    :param env: gym environment
    :param noop_max: max number of no-ops
    :param frame_skip: the frequency at which the agent experiences the game.
    :param screen_size: resize Atari frame
    :param terminal_on_life_loss: if True, then step() returns done=True whenever a life is lost.
    :param clip_reward: If True (default), the reward is clip to {-1, 0, 1} depending on its sign.
    """

    def __init__(
        self,
        env: gym.Env,
        frame_skip: int = 4,
        screen_size: int = 84,
    ):
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
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.max_reward = -np.inf

    def _on_step(self) -> bool:
        buffer = self.locals["replay_buffer"]  # type: ReplayBuffer
        infos = buffer.infos
        if not buffer.full:
            if buffer.pos == 0:
                return True
            infos = buffer.infos[: buffer.pos]

        rewards = [info[env_idx]["env_reward"] for info in infos for env_idx in range(buffer.n_envs)]
        self.max_reward = max(np.max(rewards), self.max_reward)
        self.logger.record("env/max_env_eward", self.max_reward)
        return True


class NumberCellsLogger(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.all_cells = np.zeros((0, 20, 20), dtype=np.uint8)

    def _on_step(self) -> bool:
        if self.n_calls % 10_000 == 0:
            buffer = self.locals["replay_buffer"]  # type: LGEBuffer
            observations = buffer.next_observations["observation"]
            if not buffer.full:
                observations = observations[: buffer.pos]
            observations = np.reshape(observations, (-1, 1, 84, 84))  # (N, N_ENVS, C, H, W) to (N*N_ENVS, C, H, W)
            observations = np.moveaxis(observations, 1, -1)  # (N*N_ENVS, C, H, W) to (N*N_ENVS, H, W, C)
            cells = np.zeros((observations.shape[0], 20, 20))
            for i, observation in enumerate(observations):
                cells[i] = cv2.resize(observation, (20, 20))  #  (N*N_ENVS, H, W, C) to (N*N_ENVS, 20, 20)
            cells = cells / 255 * 12  # [0, d]
            cells = (np.floor(cells) * 255 / 12).astype(np.uint8)
            self.all_cells = np.concatenate((self.all_cells, cells))
            self.all_cells = np.unique(self.all_cells, axis=0)
            self.logger.record("env/nb_cells", len(self.all_cells))
        return True
