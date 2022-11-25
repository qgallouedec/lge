from typing import Any, Dict, Mapping, Optional, Tuple, Union

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


class BitFlippingEnv(gym.Env):
    """
    Simple bit flipping env, useful to test HER.

    The goal is to flip all the bits to get a vector of ones. In the continuous variant, if the ith action component has a
    value > 0, then the ith bit will be flipped.

    :param n_bits: Number of bits to flip
    :param action_type: "discrete" or "continuous"
    :param max_steps: Max number of steps, by default, equal to n_bits
    :param discrete_obs_space: Whether to use the discrete observation version or not, by default, it uses the
        ``MultiBinary`` one
    """

    def __init__(
        self,
        n_bits: int = 10,
        action_type: str = "discrete",
        observation_type: str = "discrete",
        max_steps: Optional[int] = None,
        channel_first: bool = True,
    ) -> None:
        super().__init__()
        # The achieved goal is determined by the current state
        # here, it is a special where they are equal
        self.observation_type = observation_type
        if self.observation_type == "discrete":
            # In the discrete case, the agent act on the binary representation of the observation
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Discrete(2**n_bits),
                    "goal": spaces.Discrete(2**n_bits),
                }
            )
        elif self.observation_type in ["image_channel_first", "image_channel_last"]:
            # When using image as input, one image contains the bits 0 -> 0, 1 -> 255 and the rest is filled with zeros
            # Shape of the observation when using image space
            channel_first = self.observation_type == "image_channel_first"
            image_shape = (1, 36, 36) if channel_first else (36, 36, 1)
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0,
                        high=255,
                        shape=image_shape,
                        dtype=np.uint8,
                    ),
                    "goal": spaces.Box(
                        low=0,
                        high=255,
                        shape=image_shape,
                        dtype=np.uint8,
                    ),
                }
            )
        elif self.observation_type == "mulbinary":
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.MultiBinary(n_bits),
                    "goal": spaces.MultiBinary(n_bits),
                }
            )
        else:
            raise ValueError("Wrong observation type")

        self.action_type = action_type
        if self.action_type == "discrete":
            self.action_space = spaces.Discrete(n_bits)
        elif self.action_type == "continuous":
            self.action_space = spaces.Box(-1, 1, shape=(n_bits,), dtype=np.float32)
        else:
            raise ValueError("Wrong action type")

        self.state = np.zeros((n_bits,), dtype=np.uint8)
        self.goal = np.ones((n_bits,), dtype=np.uint8)
        self.n_bits = n_bits
        self.max_steps = max_steps if max_steps is not None else n_bits
        self.current_step = 0

    def state_to_obs(self, state: np.ndarray) -> Union[int, np.ndarray]:
        """
        Convert to discrete space if needed.
        """
        if self.observation_type == "discrete":
            # The internal state is the binary representation of the observed one
            # [0, 1, 0, 1] -> 0*8 + 1*4 + 0*2 + 1*1
            return np.packbits(state)[0]  # TODO: Ends with 8... needs to be fixed
        elif self.observation_type in ["image_channel_first", "image_channel_last"]:
            # [0, 1, 0, 1] -> [[[  0], [255], [  0], [255], [  0], [  0], [  0]],
            #                  [[  0], [  0], [  0], [  0], [  0], [  0], [  0]],
            #                  [[  0], [  0], [  0], [  0], [  0], [  0], [  0]],
            #                  [[  0], [  0], [  0], [  0], [  0], [  0], [  0]]]

            size = np.prod(self.observation_space.shape)
            image = np.concatenate((state * 255, np.zeros(size - len(state))))
            return image.reshape(self.observation_space.shape).astype(np.uint8)
        elif self.observation_type == "mulbinary":
            return state

    def obs_to_state(self, obs: Union[int, np.ndarray]) -> np.ndarray:
        """
        Convert obs to state.
        """
        if self.observation_type == "discrete":
            # The internal state is the binary representation of the observed one
            # 0*8 + 1*4 + 0*2 + 1*1 -> [0, 1, 0, 1]
            return np.unpackbits(np.array(obs).astype(np.uint8))
        elif self.observation_type in ["image_channel_first", "image_channel_last"]:
            # [[[  0], [255], [  0], [255], [  0], [  0], [  0]], -> [0, 1, 0, 1]
            #  [[  0], [  0], [  0], [  0], [  0], [  0], [  0]],
            #  [[  0], [  0], [  0], [  0], [  0], [  0], [  0]],
            #  [[  0], [  0], [  0], [  0], [  0], [  0], [  0]]]
            return obs.flatten()[: len(self.state)] / 255

        elif self.observation_type == "mulbinary":
            return obs

    def _get_obs(self) -> Union[Dict[str, int], Dict[str, np.ndarray]]:
        """
        Helper to create the observation.
        """
        return {
            "observation": self.state_to_obs(self.state.copy()),
            "goal": self.state_to_obs(self.goal.copy()),
        }

    def reset(self) -> Union[Dict[str, int], Dict[str, np.ndarray]]:
        self.current_step = 0
        self.state = np.random.randint(0, 2, (self.n_bits,), dtype=np.uint8)
        return self._get_obs()

    def step(self, action: Union[np.ndarray, int]):
        if self.action_type == "discrete":
            self.state[action] = 1 - self.state[action]
        elif self.action_type == "continuous":
            self.state[action > 0] = 1 - self.state[action > 0]
        obs = self._get_obs()
        is_succcess = (self.state == self.goal).all()
        reward = float(is_succcess) - 1
        done = is_succcess
        self.current_step += 1
        # Episode terminate when we reached the goal or the max number of steps
        info = {"is_success": is_succcess}
        done = done or self.current_step >= self.max_steps
        return obs, reward, done, info
