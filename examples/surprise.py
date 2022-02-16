import gym
import panda_gym
from go_explore.common.wrappers import UnGoalWrapper
from go_explore.surprise import SurpriseMotivation, TransitionModelLearner
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
env = VecNormalize(env, norm_reward=False)

surprise_motivation = SurpriseMotivation(
    obs_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], eta_0=1.0, hidden_size=32
)
model = SAC("MlpPolicy", env, verbose=1)
learner = TransitionModelLearner(
    transition_model=surprise_motivation.transition_model,
    buffer=model.replay_buffer,
    train_freq=100,
    grad_step=200,
    weight_decay=1e-7,
    lr=1e-4,
    batch_size=64,
)
model.learn(10000, eval_env=env, callback=[learner], reward_modifier=surprise_motivation, use_random_action=True)
