import os

import gym
import panda_gym
from go_explore.ICM import ICM
from go_explore.wrapper import UnGoalWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def train():
    beta = 0.5
    scaling_factor = 100
    lmbda = 0.01
    feature_dim = 128
    hidden_dim = 128

    env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_reward=100)

    icm = ICM(
        beta=beta,
        scaling_factor=scaling_factor,
        lmbda=lmbda,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
    )

    model = SAC("MlpPolicy", env, actor_loss_modifier=icm, reward_modifier=icm, verbose=1)

    i = 0
    path = "./results/" + str(i)

    while os.path.isdir(path):
        i += 1
        path = "./results/" + str(i)
    os.mkdir(path)
    model.learn(30000, eval_env=env, eval_freq=2000, n_eval_episodes=10, eval_log_path=path)


if __name__ == "__main__":
    train()
