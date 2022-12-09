from stable_baselines3.common.atari_wrappers import AtariWrapper


class RAMtoInfoWrapper(AtariWrapper):
    def step(self, action):
        obs, reward, done, info = super().step(action)
        info["ram"] = self.unwrapped.ale.getRAM()
        return obs, reward, done, info
