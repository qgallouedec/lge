import panda_gym
import gym
from go_explore.common.callbacks import LogNbCellsCallback, SaveNbCellsCallback
from go_explore.common.wrappers import EpisodeStartWrapper
from go_explore.go_explore.archive import ArchiveBuffer
from go_explore.go_explore.cell_computers import PandaObjectCellComputer
from go_explore.icm import ICM
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv(4 * [lambda: EpisodeStartWrapper(gym.make("PandaNoTask-v0", nb_objects=1))])
env = VecNormalize(env, norm_reward=False)
icm = ICM(
    scaling_factor=1.0,
    actor_loss_coef=10,
    inverse_loss_coef=1.0,
    forward_loss_coef=0.1,
    obs_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    feature_dim=64,
    hidden_dim=256,
)
model = SAC(
    "MlpPolicy",
    env,
    actor_loss_modifier=icm,
    replay_buffer_class=ArchiveBuffer,
    replay_buffer_kwargs={"cell_computer": PandaObjectCellComputer()},
    verbose=1
)
logger = LogNbCellsCallback(model.replay_buffer)
saver = SaveNbCellsCallback(model.replay_buffer, save_fq=500)
model.learn(50000, callback=[saver, logger], reward_modifier=icm)
print(saver.nb_cells)
