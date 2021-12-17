import gym
import panda_gym
from go_explore.common.callbacks import SaveNbCellsCallback
from go_explore.go_explore.cell_computers import PandaCellComputer
from go_explore.go_explore.go_explore import GoExplore

env = gym.make("PandaNoTask-v0")
ge = GoExplore(env, PandaCellComputer(), count_pow=6, subgoal_horizon=10, done_delay=8)
saver = SaveNbCellsCallback(ge.archive, save_fq=500)
ge.exploration(50000, callback=saver)
print(saver.nb_cells)
