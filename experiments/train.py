import os

import gym
import numpy as np
import panda_gym
from go_explore.simhash import SimHashWrapper
from go_explore.wrapper import UnGoalWrapper
from stable_baselines3 import TD3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


# Define an objective function to be minimized.
def train():
    beta = 0.01
    granularity = 2048

    env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_reward=100)
    env = SimHashWrapper(env, granularity=granularity, beta=beta)

    action_noise_cls = OrnsteinUhlenbeckActionNoise
    action_noise = action_noise_cls(mean=np.zeros(env.action_space.shape), sigma=np.ones(env.action_space.shape) * 0.5)

    rewards = []
    for _ in range(1):

        model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
        eval_env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_reward=100)

        # sum_reward = 0
        # for _ in range(50):
        #     obs = eval_env.reset()
        #     done = False
        #     while not done:
        #         action = model.predict(obs)[0]
        #         obs, reward, done, info = eval_env.step(action)
        #         sum_reward += reward
        # rewards.append(sum_reward)

        i=0
        path = "./results/" + str(i)

        while os.path.isdir(path):
            i += 1
            path = "./results/" + str(i)
        os.mkdir(path)
        model.learn(50000, eval_env=eval_env, eval_freq=2000, n_eval_episodes=10, eval_log_path="./results/" + str(i))


    return np.median(rewards)


if __name__ == "__main__":
    print(train())


#  for MoutainCar {'beta': 0.013905647986691212, 'granularity': 64}. Best is trial 3 with value: 4641.87744140625.
