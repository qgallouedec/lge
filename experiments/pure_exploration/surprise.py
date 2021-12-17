import gym
import panda_gym
from go_explore.common.callbacks import SaveNbCellsCallback
from go_explore.common.wrappers import EpisodeStartWrapper
from go_explore.go_explore.archive import ArchiveBuffer
from go_explore.go_explore.cell_computers import PandaCellComputer
from go_explore.surprise import SurpriseMotivation, TransitionModelLearner
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv(4 * [lambda: EpisodeStartWrapper(gym.make("PandaNoTask-v0"))])
env = VecNormalize(env, norm_reward=False)
surprise_motivation = SurpriseMotivation(
    obs_size=env.observation_space.shape[0],
    action_size=env.action_space.shape[0],
    eta_0=5.0,
    hidden_size=256,
)
model = SAC("MlpPolicy", env, replay_buffer_class=ArchiveBuffer, replay_buffer_kwargs={"cell_computer": PandaCellComputer()})
learner = TransitionModelLearner(
    transition_model=surprise_motivation.transition_model,
    buffer=model.replay_buffer,
    train_freq=10,
    grad_step=1,
    weight_decay=1e-6,
    lr=1e-5,
    batch_size=128,
)
saver = SaveNbCellsCallback(model.replay_buffer, save_fq=500)
model.learn(50000, callback=[saver, learner], reward_modifier=surprise_motivation)
print(saver.nb_cells)
