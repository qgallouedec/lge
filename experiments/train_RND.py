import os

import gym
import panda_gym
from go_explore.RND import RND, PredictorLearner
from go_explore.wrapper import UnGoalWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def train():
    scaling_factor = 0.1
    out_dim = 64
    hidden_dim = 32
    train_freq = 10
    grad_step = 1000
    weight_decay = 1e-6
    lr = 1e-5
    batch_size = 64

    env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_reward=100)

    rnd = RND(scaling_factor=scaling_factor, obs_dim=env.observation_space.shape[0], out_dim=out_dim, hidden_dim=hidden_dim)

    model = SAC("MlpPolicy", env, reward_modifier=rnd, verbose=1)
    cb = PredictorLearner(
        predictor=rnd.predictor,
        target=rnd.target,
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
    model.learn(10000, callback=cb, eval_env=env, eval_freq=2000, n_eval_episodes=10, eval_log_path=path)


if __name__ == "__main__":
    train()
