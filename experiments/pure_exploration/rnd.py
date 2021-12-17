import gym
import panda_gym
from go_explore.common.callbacks import SaveNbCellsCallback
from go_explore.common.wrappers import EpisodeStartWrapper
from go_explore.go_explore.archive import ArchiveBuffer
from go_explore.go_explore.cell_computers import PandaCellComputer
from go_explore.rnd import RND, PredictorLearner
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv(4 * [lambda: EpisodeStartWrapper(gym.make("PandaNoTask-v0"))])
env = VecNormalize(env, norm_reward=False)
rnd = RND(scaling_factor=100.0, obs_dim=env.observation_space.shape[0], out_dim=64, hidden_dim=64)
model = SAC(
    "MlpPolicy",
    env,
    replay_buffer_class=ArchiveBuffer,
    replay_buffer_kwargs={"cell_computer": PandaCellComputer()},
)
learner = PredictorLearner(
    predictor=rnd.predictor,
    target=rnd.target,
    buffer=model.replay_buffer,
    train_freq=100,
    grad_step=10,
    weight_decay=1e-4,
    lr=0.001,
    batch_size=64,
)
saver = SaveNbCellsCallback(model.replay_buffer, save_fq=500)
model.learn(50000, callback=[saver, learner], reward_modifier=rnd)
print(saver.nb_cells)
