import os

import gym
from go_explore.simhash import SimHashWrapper
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

for i in range(10):
    env = DummyVecEnv([lambda: gym.make("MountainCarContinuous-v0")])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_reward=100)
    env = SimHashWrapper(env, granularity=128, beta=0)

    eval_env = DummyVecEnv([lambda: gym.make("MountainCarContinuous-v0")])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_reward=100)
    eval_env = SimHashWrapper(eval_env, granularity=128, beta=0)

    model = TD3(env=env, policy="MlpPolicy")
    os.mkdir("./results/" + str(i))
    model.learn(30000, eval_env=eval_env, eval_freq=2000, n_eval_episodes=10, eval_log_path="./results/" + str(i))
