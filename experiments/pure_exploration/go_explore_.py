import gym
import panda_gym
from go_explore.common.callbacks import SaveNbCellsCallback
from go_explore.go_explore.cell_computers import PandaObjectCellComputer
from go_explore.go_explore.go_explore import GoExplore

env = gym.make("PandaNoTask-v0", nb_objects=1)
ge = GoExplore(env, PandaObjectCellComputer(), count_pow=0.5, subgoal_horizon=7, done_delay=16, gradient_steps=500, verbose=1)
saver = SaveNbCellsCallback(ge.archive, save_fq=1000)
ge.exploration(20000, callback=saver)
print(saver.nb_cells)
