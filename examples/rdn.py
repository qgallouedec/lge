import gym
import panda_gym
from go_explore.common.wrappers import UnGoalWrapper
from go_explore.rnd import RND, PredictorLearner
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

scaling_factor = 00
out_dim = 64
hidden_dim = 128
train_freq = 10
grad_step = 100
weight_decay = 1e-2
lr = 1e-5
batch_size = 512

env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
env = VecNormalize(env, norm_reward=False)

rnd = RND(scaling_factor=scaling_factor, obs_dim=env.observation_space.shape[0], out_dim=out_dim, hidden_dim=hidden_dim)

model = SAC("MlpPolicy", env, verbose=1)
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

model.learn(10000, callback=cb, eval_env=env, eval_freq=1000, n_eval_episodes=10, reward_modifier=rnd)
