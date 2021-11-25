import gym
import panda_gym
from go_explore.ICM import ICM
from go_explore.wrapper import UnGoalWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_reward=100)

icm = ICM(beta=0.5, scaling_factor=0, lmbda=1, obs_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])

model = SAC("MlpPolicy", env, actor_loss_modifier=icm, reward_modifier=icm, verbose=1)

model.learn(30000)
