import gym
import numpy as np
import panda_gym
from go_explore.common.wrappers import UnGoalWrapper
from go_explore.go_explore.cell_computers import PandaCellComputer
from go_explore.go_explore.go_explore import GoExplore
from stable_baselines3 import SAC

env = gym.make("PandaReach-v2")
env = UnGoalWrapper(env)
cell_computer = PandaCellComputer()

go_explore = GoExplore(
    env=env,
    explore_model_cls=SAC,
    cell_computer=cell_computer,
    subgoal_horizon=10,
    count_pow=4,
)

n = 10000
go_explore.exploration(n)


def task(obs):
    result = (obs[..., 5] > 0.2).astype(np.float32)
    return result


model = go_explore.learn_task(SAC, task, n, 256)


sum_reward = 0
for _ in range(50):
    obs = env.reset()
    done = False
    while not done:
        action = model.predict(obs)[0]
        obs, _, done, info = env.step(action)
        reward = task(obs)
        sum_reward += reward
print(sum_reward)
