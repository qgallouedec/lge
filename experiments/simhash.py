import gym
import panda_gym
from go_explore.simhash import SimHashMotivation
from go_explore.wrapper import UnGoalWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_reward=100)

simhash = SimHashMotivation(obs_dim=env.observation_space.shape[0], granularity=128, beta=1)

model = SAC("MlpPolicy", env, reward_modifier=simhash, verbose=1)

model.learn(30000)
