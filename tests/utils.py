from typing import Any, Dict, Mapping, Tuple, Union

import gym
import numpy as np
from gym import spaces

Observation = Dict[str, Union[np.ndarray, int]]
Action = Union[np.ndarray, int]


class DummyEnv(gym.Env):
    def __init__(self, observation_space: spaces.Dict, action_space: gym.Space) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        return self.observation_space.sample()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Mapping[str, Any]]:
        return self.observation_space.sample(), 0.0, False, {}
