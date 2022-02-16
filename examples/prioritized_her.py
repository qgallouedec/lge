
import gym
import panda_gym
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from go_explore.go_explore.archive import PrioritizedHerDictReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

from go_explore.go_explore.cell_computers import PandaCellComputer
from go_explore.common.wrappers import EpisodeStartWrapper
env = gym.make("PandaReach-v2")
env = EpisodeStartWrapper(env)
env = DummyVecEnv([lambda: env])
buf = PrioritizedHerDictReplayBuffer(env, 600, PandaCellComputer())

for _ in range(10):
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, info  = env.step(action)
        buf.add(obs, next_obs, action, reward, done, info)
        obs = next_obs

print(buf.sample(16, None))