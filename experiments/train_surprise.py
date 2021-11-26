import os

import gym
import panda_gym
from go_explore.surprise import SurpriseMotivation, TransitionModelLearner
from go_explore.wrapper import UnGoalWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


def train():
    eta = 0.04
    hidden_size = 32
    train_freq = 5
    grad_step = 50
    weight_decay = 1e-6
    lr = 0.001
    batch_size = 64

    env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_reward=100)

    surprise_motivation = SurpriseMotivation(
        obs_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], eta=eta, hidden_size=hidden_size
    )

    model = SAC("MlpPolicy", env, reward_modifier=surprise_motivation, verbose=1)
    cb = TransitionModelLearner(
        transition_model=surprise_motivation.transition_model,
        buffer=model.replay_buffer,
        train_freq=train_freq,
        grad_step=grad_step,
        weight_decay=weight_decay,
        lr=lr,
        batch_size=batch_size,
    )

    i = 0
    path = "./results/" + str(i)

    while os.path.isdir(path):
        i += 1
        path = "./results/" + str(i)
    os.mkdir(path)
    model.learn(30000, callback=cb, eval_env=env, eval_freq=2000, n_eval_episodes=10, eval_log_path=path)


if __name__ == "__main__":
    train()
