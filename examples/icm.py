import gym
import panda_gym
from go_explore.icm.icm import ICM
from go_explore.common.wrappers import UnGoalWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

scaling_factor = 1.0
actor_loss_coef = 1.0
inverse_loss_coef = 0.5
forward_loss_coef = 0.5
feature_dim = 16
hidden_dim = 16

env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
env = VecNormalize(env, norm_reward=False)
icm = ICM(
    scaling_factor=scaling_factor,
    actor_loss_coef=actor_loss_coef,
    inverse_loss_coef=inverse_loss_coef,
    forward_loss_coef=forward_loss_coef,
    obs_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    feature_dim=feature_dim,
    hidden_dim=hidden_dim,
)
model = SAC("MlpPolicy", env)  # , actor_loss_modifier=icm)
model.learn(10000, eval_env=env, eval_freq=100, n_eval_episodes=10)  # , reward_modifier=icm)
