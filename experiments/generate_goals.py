"""Generate a numpy file containing goals for Panda without object.
"""

import gym
import numpy as np
import panda_gym
from go_explore.wrapper import UnGoalWrapper

env = gym.make("PandaReach-v1")
env = UnGoalWrapper(env)
obss = []

for _ in range(1000):
    env.reset()
    for _ in range(50):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
    obss.append(obs)
obss = np.array(obss)

with open('goals.npy', 'wb') as f:
    np.save(f, obss)

with open('goals.npy', 'rb') as f:
    goals = np.load(f)

print(goals[..., 3:6])
