import gym
import panda_gym
from go_explore.common.wrappers import UnGoalWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
env = VecNormalize(env, norm_reward=False)
model = SAC("MlpPolicy", env)
model.learn(10000, eval_env=env, eval_freq=100, n_eval_episodes=10)
