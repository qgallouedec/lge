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
