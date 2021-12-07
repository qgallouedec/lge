from typing import Any, Dict, Tuple

import gym
import gym.spaces
import numpy as np


class EpisodeStartWrapper(gym.Wrapper):
    """Add {"episode_start": True} in info when episode starts."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reset(self) -> Any:
        self.episode_starts = True
        return super().reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        next_obs, reward, done, info = super().step(action)
        if self.episode_starts:
            info["episode_start"] = True
            self.episode_starts = False
        return next_obs, reward, done, info


class UnGoalWrapper(gym.Wrapper):
    """
    Observation wrapper that flattens the observation. ``achieved_goal`` is removed.

    :param env: The environment to be wrapped
    """

    def __init__(self, env: gym.GoalEnv) -> None:
        super(UnGoalWrapper, self).__init__(env)
        env.observation_space.spaces.pop("achieved_goal")
        self.observation_space = gym.spaces.flatten_space(env.observation_space)

    def reset(self) -> Any:
        observation = super().reset()
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        observation.pop("achieved_goal")
        observation = gym.spaces.flatten(self.env.observation_space, observation)
        return observation

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any]) -> None:
        raise NotImplementedError("This method is not accessible, since the environment is not a GoalEnv.")
