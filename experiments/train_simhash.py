import os

import gym
import panda_gym
from go_explore.simhash import SimHashMotivation
from go_explore.wrapper import UnGoalWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


def train():
    granularity = 128
    beta = 1

    env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_reward=100)

    simhash = SimHashMotivation(
        obs_dim=env.observation_space.shape[0],
        granularity=granularity,
        beta=beta,
    )

    model = SAC("MlpPolicy", env, reward_modifier=simhash, verbose=1)

    i = 0
    path = "./results/" + str(i)

    while os.path.isdir(path):
        i += 1
        path = "./results/" + str(i)
    os.mkdir(path)
    model.learn(30000, eval_freq=2000, n_eval_episodes=10, eval_log_path=path)


if __name__ == "__main__":
    train()
