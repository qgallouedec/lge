import gym
import panda_gym
from go_explore.common.callbacks import SaveNbCellsCallback
from go_explore.common.wrappers import EpisodeStartWrapper
from go_explore.go_explore.archive import ArchiveBuffer
from go_explore.go_explore.cell_computers import PandaCellComputer
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv(4 * [lambda: EpisodeStartWrapper(gym.make("PandaNoTask-v0"))])
model = SAC("MlpPolicy", env, replay_buffer_class=ArchiveBuffer, replay_buffer_kwargs={"cell_computer": PandaCellComputer()})
saver = SaveNbCellsCallback(model.replay_buffer, save_fq=500)
model.learn(50000, callback=[saver])
print(saver.nb_cells)
