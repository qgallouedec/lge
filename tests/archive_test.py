from collections import OrderedDict
from typing import Any, Dict, Optional, Union

import numpy as np
import pytest
import torch
from gym import GoalEnv, spaces
from gym.envs.registration import EnvSpec
from stable_baselines3 import DDPG, DQN, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.type_aliases import GymStepReturn
from stable_baselines3.common.utils import get_device
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

from lge.buffer import LGEBuffer
from lge.inverse_model import LinearInverseModel
from lge.utils import index

device = get_device()


class BitFlippingEnv(GoalEnv):
    """
    Simple bit flipping env, useful to test HER.
    The goal is to flip all the bits to get a vector of ones.
    In the continuous variant, if the ith action component has a value > 0,
    then the ith bit will be flipped.

    :param n_bits: Number of bits to flip
    :param continuous: Whether to use the continuous actions version or not,
        by default, it uses the discrete one
    :param max_steps: Max number of steps, by default, equal to n_bits
    :param discrete_obs_space: Whether to use the discrete observation
        version or not, by default, it uses the ``MultiBinary`` one
    :param image_obs_space: Use image as input instead of the ``MultiBinary`` one.
    :param channel_first: Whether to use channel-first or last image.
    """

    spec = EnvSpec("BitFlippingEnv-v0")

    def __init__(
        self,
        n_bits: int = 10,
        continuous: bool = False,
        max_steps: Optional[int] = None,
        discrete_obs_space: bool = False,
        image_obs_space: bool = False,
        channel_first: bool = True,
    ) -> None:
        super().__init__()
        # Shape of the observation when using image space
        self.image_shape = (1, 36, 36) if channel_first else (36, 36, 1)
        # The achieved goal is determined by the current state
        # here, it is a special where they are equal
        if discrete_obs_space:
            # In the discrete case, the agent act on the binary
            # representation of the observation
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Discrete(2**n_bits),
                    "goal": spaces.Discrete(2**n_bits),
                }
            )
        elif image_obs_space:
            # When using image as input,
            # one image contains the bits 0 -> 0, 1 -> 255
            # and the rest is filled with zeros
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0,
                        high=255,
                        shape=self.image_shape,
                        dtype=np.uint8,
                    ),
                    "goal": spaces.Box(
                        low=0,
                        high=255,
                        shape=self.image_shape,
                        dtype=np.uint8,
                    ),
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.MultiBinary(n_bits),
                    "goal": spaces.MultiBinary(n_bits),
                }
            )

        self.obs_space = spaces.MultiBinary(n_bits)

        if continuous:
            self.action_space = spaces.Box(-1, 1, shape=(n_bits,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(n_bits)
        self.continuous = continuous
        self.discrete_obs_space = discrete_obs_space
        self.image_obs_space = image_obs_space
        self.state = None
        self.goal = np.ones((n_bits,))
        if max_steps is None:
            max_steps = n_bits
        self.max_steps = max_steps
        self.current_step = 0

    def seed(self, seed: int) -> None:
        self.obs_space.seed(seed)

    def convert_if_needed(self, state: np.ndarray) -> Union[int, np.ndarray]:
        """
        Convert to discrete space if needed.

        :param state:
        :return:
        """
        if self.discrete_obs_space:
            # The internal state is the binary representation of the
            # observed one
            return int(sum([state[i] * 2**i for i in range(len(state))]))

        if self.image_obs_space:
            size = np.prod(self.image_shape)
            image = np.concatenate((state * 255, np.zeros(size - len(state), dtype=np.uint8)))
            return image.reshape(self.image_shape).astype(np.uint8)
        return state

    def convert_to_bit_vector(self, state: Union[int, np.ndarray], batch_size: int) -> np.ndarray:
        """
        Convert to bit vector if needed.

        :param state:
        :param batch_size:
        :return:
        """
        if batch_size == 0:
            return state
        # Convert back to bit vector
        elif isinstance(state, int):
            state = np.array(state).reshape(batch_size, -1)
            # Convert to binary representation
            state = (((state[:, :] & (1 << np.arange(len(self.state))))) > 0).astype(int)
        elif self.image_obs_space:
            state = state.reshape(batch_size, -1)[:, : len(self.state)] / 255
        else:
            state = np.array(state).reshape(batch_size, -1)

        return state

    def _get_obs(self) -> Dict[str, Union[int, np.ndarray]]:
        """
        Helper to create the observation.

        :return: The current observation.
        """
        return OrderedDict(
            [
                ("observation", self.convert_if_needed(self.state.copy())),
                ("goal", self.convert_if_needed(self.goal.copy())),
            ]
        )

    def reset(self) -> Dict[str, Union[int, np.ndarray]]:
        self.current_step = 0
        self.state = self.obs_space.sample()
        return self._get_obs()

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        if self.continuous:
            self.state[action > 0] = 1 - self.state[action > 0]
        else:
            self.state[action] = 1 - self.state[action]
        obs = self._get_obs()
        reward = float(self.compute_reward(obs["observation"], obs["goal"], None))
        done = reward == 0
        self.current_step += 1
        # Episode terminate when we reached the goal or the max number of steps
        info = {"is_success": done}
        done = done or self.current_step >= self.max_steps
        return obs, reward, done, info

    def compute_reward(
        self, observation: Union[int, np.ndarray], goal: Union[int, np.ndarray], _info: Optional[Dict[str, Any]]
    ) -> np.float32:
        # As we are using a vectorized version, we need to keep track of the `batch_size`
        if isinstance(observation, int):
            batch_size = 1
        elif self.image_obs_space:
            batch_size = observation.shape[0] if len(observation.shape) > 3 else 1
        else:
            batch_size = observation.shape[0] if len(observation.shape) > 1 else 1

        goal = self.convert_to_bit_vector(goal, batch_size)
        observation = self.convert_to_bit_vector(observation, batch_size)

        # Deceptive reward: it is positive only when the goal is achieved
        # Here we are using a vectorized version
        distance = np.linalg.norm(observation - goal, axis=-1)
        return -(distance > 0).astype(np.float32)

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if mode == "rgb_array":
            return self.state.copy()
        print(self.state)

    def close(self) -> None:
        pass


@pytest.mark.parametrize("n_envs", [1, 2])
@pytest.mark.parametrize("model_class", [SAC, TD3, DDPG, DQN])
def test_her(n_envs, model_class):
    # Test Hindsight Experience Replay in LGEBuffer.
    def env_fn():
        return BitFlippingEnv(n_bits=10, continuous=not (model_class == DQN))

    env = make_vec_env(env_fn, n_envs)
    n = env.action_space.n if type(env.action_space) is spaces.Discrete else env.action_space.shape[0]
    inverse_model = LinearInverseModel(env.observation_space["observation"].shape[0], n, latent_size=2).to(device)

    model = model_class(
        "MultiInputPolicy",
        env,
        replay_buffer_class=LGEBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=2,
            goal_selection_strategy="future",
            inverse_model=inverse_model,
        ),
        train_freq=4,
        gradient_steps=n_envs,
        policy_kwargs=dict(net_arch=[64]),
        learning_starts=100,
        buffer_size=int(2e4),
    )

    model.learn(total_timesteps=150)
    evaluate_policy(model, Monitor(env_fn()))


@pytest.mark.parametrize("model_class", [TD3, DQN])
def test_multiprocessing(model_class):
    def env_fn():
        return BitFlippingEnv(n_bits=10, continuous=not (model_class == DQN))

    env = make_vec_env(env_fn, n_envs=2)
    n = env.action_space.n if type(env.action_space) is spaces.Discrete else env.action_space.shape[0]
    inverse_model = LinearInverseModel(env.observation_space["observation"].shape[0], n, latent_size=2).to(device)

    model = model_class(
        "MultiInputPolicy",
        env,
        replay_buffer_class=LGEBuffer,
        replay_buffer_kwargs=dict(inverse_model=inverse_model),
        train_freq=4,
    )
    model.learn(total_timesteps=150)


@pytest.mark.parametrize(
    "goal_selection_strategy",
    [
        "final",
        "episode",
        "future",
        GoalSelectionStrategy.FINAL,
        GoalSelectionStrategy.EPISODE,
        GoalSelectionStrategy.FUTURE,
    ],
)
def test_goal_selection_strategy_with_model(goal_selection_strategy):
    """
    Test different goal strategies.
    """
    # Offline sampling is not compatible with multiprocessing
    n_envs = 2

    def env_fn():
        return BitFlippingEnv(n_bits=10, continuous=True)

    env = make_vec_env(env_fn, n_envs)
    n = env.action_space.n if type(env.action_space) is spaces.Discrete else env.action_space.shape[0]
    inverse_model = LinearInverseModel(env.observation_space["observation"].shape[0], n, latent_size=2).to(device)

    normal_action_noise = NormalActionNoise(np.zeros(1), 0.1 * np.ones(1))

    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=LGEBuffer,
        replay_buffer_kwargs=dict(
            goal_selection_strategy=goal_selection_strategy,
            n_sampled_goal=2,
            inverse_model=inverse_model,
        ),
        train_freq=4,
        gradient_steps=n_envs,
        policy_kwargs=dict(net_arch=[64]),
        learning_starts=100,
        buffer_size=int(1e5),
        action_noise=normal_action_noise,
    )
    assert model.action_noise is not None
    model.learn(total_timesteps=150)


@pytest.mark.parametrize(
    "goal_selection_strategy",
    [
        "final",
        "episode",
        "future",
        GoalSelectionStrategy.FINAL,
        GoalSelectionStrategy.EPISODE,
        GoalSelectionStrategy.FUTURE,
    ],
)
def test_goal_selection_strategy(goal_selection_strategy):
    # Test different goal strategies.
    def env_fn():
        return BitFlippingEnv(n_bits=2, continuous=True)

    env = make_vec_env(env_fn)
    n = env.action_space.n if type(env.action_space) is spaces.Discrete else env.action_space.shape[0]
    inverse_model = LinearInverseModel(env.observation_space["observation"].shape[0], n, latent_size=2).to(device)

    buffer = LGEBuffer(
        100,
        env.observation_space,
        env.action_space,
        env,
        inverse_model,
        goal_selection_strategy=goal_selection_strategy,
        n_sampled_goal=np.inf,  # All goals are virtual
        device=device,
    )

    observations = np.array([[[0, 0]], [[1, 1]], [[2, 2]], [[3, 3]], [[4, 4]], [[5, 5]], [[6, 6]], [[7, 7]]])
    goals = np.array([[[8, 8]], [[8, 8]], [[8, 8]], [[8, 8]], [[8, 8]], [[8, 8]], [[8, 8]], [[8, 8]]])
    for i in range(7):
        buffer.add(
            obs={"observation": observations[i], "goal": goals[i]},
            next_obs={
                "observation": observations[i + 1],
                "goal": goals[i + 1],
            },
            action=np.array([[0.0]]),
            reward=np.array([-1.0]),
            done=np.array([i == 6]),
            infos=[{}],
        )
    samples = buffer.sample(5)
    if goal_selection_strategy in ["future", GoalSelectionStrategy.FUTURE]:
        assert (samples.observations["observation"] < samples.observations["goal"]).all()
    elif goal_selection_strategy in ["episode", GoalSelectionStrategy.EPISODE]:
        assert np.all([index(goal.cpu().numpy(), observations) is not None for goal in samples.observations["goal"]])
    elif goal_selection_strategy in ["final", GoalSelectionStrategy.FINAL]:
        assert np.all(samples.observations["goal"].cpu().numpy() == np.array([[7, 7], [7, 7], [7, 7], [7, 7], [7, 7]]))


def test_full_replay_buffer():
    # Test if HER works correctly with a full replay buffer when using online sampling.
    # It should not sample the current episode which is not finished.
    n_bits = 10
    n_envs = 2

    def env_fn():
        return BitFlippingEnv(n_bits, continuous=True)

    env = make_vec_env(env_fn, n_envs)
    n = env.action_space.n if type(env.action_space) is spaces.Discrete else env.action_space.shape[0]
    inverse_model = LinearInverseModel(env.observation_space["observation"].shape[0], n, latent_size=2).to(device)

    # use small buffer size to get the buffer full
    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=LGEBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=2,
            goal_selection_strategy="future",
            inverse_model=inverse_model,
        ),
        gradient_steps=1,
        train_freq=4,
        policy_kwargs=dict(net_arch=[64]),
        learning_starts=n_bits * n_envs,
        buffer_size=20 * n_envs,
        verbose=1,
        seed=757,
    )
    model.learn(total_timesteps=100)


def test_trajectory_manager():
    def env_fn():
        return BitFlippingEnv(1, continuous=True)

    env = make_vec_env(env_fn, 1)
    inverse_model = LinearInverseModel(obs_size=2, action_size=1, latent_size=2).to(device)
    # Useless for this test
    action = np.array([[0], [0]])
    reward = np.array([0, 0])
    infos = [{}, {}]
    goal = np.array([[0, 0], [0, 0]])

    space = spaces.Box(-10, 10, (2,))
    buffer = LGEBuffer(
        buffer_size=100,
        observation_space=spaces.Dict({"observation": space, "goal": space}),
        action_space=space,
        env=env,
        inverse_model=inverse_model,
        n_envs=2,
        device=device,
    )
    trajectories = np.array(
        [
            [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]],
            [[0, 0], [1, 0], [1, 1], [1, 2], [1, 3], [0, 3], [0, 4]],
        ]
    )
    for i in range(6):
        buffer.add(
            obs={"observation": trajectories[:, i], "goal": goal},
            next_obs={
                "observation": trajectories[:, i + 1],
                "goal": goal,
            },
            action=action,
            reward=reward,
            done=np.ones(2) * (i == 6),
            infos=infos,
        )
    buffer.recompute_embeddings()
    sampled_trajectories = [
        list(buffer.sample_trajectory()[0].astype(int).tolist()) for _ in range(100)
    ]  # list convinient to compare
    possible_trajectories = [
        [[0, 1]],
        [[0, 1], [0, 2]],
        [[0, 1], [0, 2], [0, 3]],
        [[0, 1], [0, 2], [0, 3], [0, 4]],
        [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],
        [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]],
        [[1, 0]],
        [[1, 0], [1, 1]],
        [[1, 0], [1, 1], [1, 2]],
        [[1, 0], [1, 1], [1, 2], [1, 3]],
        [[1, 0], [1, 1], [1, 2], [1, 3], [0, 3]],
        [[1, 0], [1, 1], [1, 2], [1, 3], [0, 3], [0, 4]],
    ]
    # Check that all sampled trajectories are valid
    assert np.all([trajectory in possible_trajectories for trajectory in sampled_trajectories])

    # Check that all valid trajectories are sampled
    assert np.all([trajectory in sampled_trajectories for trajectory in possible_trajectories])


@pytest.mark.parametrize("goal_selection_strategy", ["final", "episode", "future"])
def test_performance_her(goal_selection_strategy):
    """
    That DQN+HER can solve BitFlippingEnv.
    It should not work when n_sampled_goal=0 (DQN alone).
    """
    # Offline sampling is not compatible with multiprocessing
    n_envs = 2

    def env_fn():
        return BitFlippingEnv(n_bits=2, continuous=False)

    env = make_vec_env(env_fn, 1)
    inverse_model = LinearInverseModel(obs_size=2, action_size=1, latent_size=2).to(device)

    model = DQN(
        "MultiInputPolicy",
        env,
        replay_buffer_class=LGEBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=5,
            goal_selection_strategy=goal_selection_strategy,
            inverse_model=inverse_model,
        ),
        verbose=1,
        learning_rate=5e-3,
        train_freq=1,
        gradient_steps=n_envs,
        learning_starts=100,
        exploration_final_eps=0.02,
        target_update_interval=500,
        seed=0,
        batch_size=32,
        buffer_size=int(1e5),
    )

    model.learn(total_timesteps=5000, log_interval=50)

    # 90% training success
    assert np.mean(model.ep_success_buffer) > 0.90


def test_sample_if_empty():
    space = spaces.Box(-10, 10, (1,))
    inverse_model = LinearInverseModel(obs_size=1, action_size=1, latent_size=2).to(device)
    buffer = LGEBuffer(
        buffer_size=100,
        observation_space=spaces.Dict({"observation": space, "goal": space}),
        action_space=spaces.Box(-10, 10, (1,)),
        env=GoalEnv(),
        inverse_model=inverse_model,
        n_envs=2,
        device=device,
    )
    trajectory, _ = buffer.sample_trajectory()
    for obs in trajectory:
        assert space.contains(obs)
