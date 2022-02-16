import gym
import panda_gym
from go_explore.common.wrappers import UnGoalWrapper
from go_explore.simhash import SimHashMotivation
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
env = VecNormalize(env, norm_reward=False)
model = SAC("MlpPolicy", env, verbose=1, batch_size=8, learning_starts=10)
simhash = SimHashMotivation(model.replay_buffer, env, granularity=4, beta=1)
model.learn(10000, reward_modifier=simhash)
