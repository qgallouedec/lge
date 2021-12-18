import gym
import panda_gym
from go_explore.common.wrappers import EpisodeStartWrapper, UnGoalWrapper
from gym import spaces


def test_ungoal_wrapper():
    env = gym.make("PandaReach-v2")
    env = UnGoalWrapper(env)
    assert type(env.observation_space) is spaces.Box
    obs = env.reset()
    assert obs.shape == (9,)  # "observation": 6, "desired_goal": 3


def test_episode_starter():
    env = gym.make("PandaReach-v2")
    env = EpisodeStartWrapper(env)
    env.reset()
    _, _, _, info = env.step(env.action_space.sample())
    assert info.get("episode_start", False)
    _, _, _, info = env.step(env.action_space.sample())
    assert not info.get("episode_start", False)
