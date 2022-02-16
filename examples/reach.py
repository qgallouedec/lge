import gym
import panda_gym
from go_explore.go_explore.cell_computers import PandaCellComputer

from go_explore.go_explore.go_explore import GoExplore

env = gym.make("PandaNoTask-v0")
ge = GoExplore(env, PandaCellComputer(), subgoal_horizon=10, verbose=1)
ge.exploration(15000, eval_freq=500, n_eval_episodes=50)
