import gym
import panda_gym
from go_explore.common.callbacks import SaveNbCellsCallback
from go_explore.common.wrappers import EpisodeStartWrapper
from go_explore.go_explore.archive import ArchiveBuffer
from go_explore.go_explore.cell_computers import PandaCellComputer
from go_explore.simhash import SimHashMotivation
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv(4 * [lambda: EpisodeStartWrapper(gym.make("PandaNoTask-v0"))])
env = VecNormalize(env, norm_reward=False)
model = SAC("MlpPolicy", env, replay_buffer_class=ArchiveBuffer, replay_buffer_kwargs={"cell_computer": PandaCellComputer()})
saver = SaveNbCellsCallback(model.replay_buffer, save_fq=500)
simhash = SimHashMotivation(model.replay_buffer, env, granularity=512, beta=10)
model.learn(50000, callback=saver, reward_modifier=simhash)
print(saver.nb_cells)
