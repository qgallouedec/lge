import numpy as np
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback


class RAMtoInfoWrapper(AtariWrapper):
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
        self.logger.record("max env eward", self.max_reward)
        return True
