import os

from go_explore.envs import PandaReachFlat
from go_explore.icm.icm import ICM
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gym
from go_explore.common.wrappers import UnGoalWrapper


def train():
    beta = 0.5
    scaling_factor = 0.36
    lmbda = 0.001
    feature_dim = 16
    hidden_dim = 8

    env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
    env = VecNormalize(env, norm_reward=False)

    icm = ICM(
        beta=beta,
        scaling_factor=scaling_factor,
        lmbda=lmbda,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
    )

    model = SAC("MlpPolicy", env, reward_modifier=icm, actor_loss_modifier=icm, verbose=1)

    i = 0
    path = "./results/" + str(i)
    while os.path.isdir(path):
        i += 1
        path = "./results/" + str(i)
    os.mkdir(path)

    model.learn(10000, eval_env=env, eval_freq=500, n_eval_episodes=10, eval_log_path=path)


if __name__ == "__main__":
    train()
