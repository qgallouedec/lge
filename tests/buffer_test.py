import numpy as np
import pytest
from gym import spaces
from stable_baselines3.common.env_util import make_vec_env

from lge.buffer import LGEBuffer
from lge.modules.ae_module import AEModule
from lge.modules.forward_module import ForwardModule
from lge.modules.inverse_module import InverseModule
from lge.utils import get_size
from tests.utils import DummyEnv

OBSERVATION_SPACES = [
    spaces.Discrete(3),
    spaces.MultiDiscrete([3, 2]),
    spaces.MultiBinary(3),
    # spaces.MultiBinary([3, 2]), # Not working so far
    spaces.Box(-2, 2, shape=(2,)),
    spaces.Box(-2, 2, shape=(2, 2)),
    spaces.Box(0, 255, shape=(36, 36, 1), dtype=np.uint8),  # BW channel last image
    spaces.Box(0, 255, shape=(36, 36, 3), dtype=np.uint8),  # RGB channel last image
    spaces.Box(0, 255, shape=(1, 36, 36), dtype=np.uint8),  # BW channel first image
    spaces.Box(0, 255, shape=(3, 36, 36), dtype=np.uint8),  # RGB channel first image
]

ACTION_SPACES = [
    spaces.Discrete(3),
    spaces.MultiDiscrete([3, 2]),
    spaces.MultiBinary(3),
    # spaces.MultiBinary([3, 2]), # Not working so far
    spaces.Box(-2, 2, shape=(2,)),
    spaces.Box(-2, 2, shape=(2, 2)),
]


@pytest.mark.parametrize("observation_space", OBSERVATION_SPACES)
@pytest.mark.parametrize("action_space", ACTION_SPACES)
@pytest.mark.parametrize("module_class", [AEModule, InverseModule, ForwardModule])
def test_add(observation_space, action_space, module_class):
    obs_size = get_size(observation_space)
    action_size = get_size(action_space)
    if module_class is AEModule:
        module = AEModule(obs_size)
    elif module_class is InverseModule:
        module = InverseModule(obs_size, action_size)
    elif module_class is ForwardModule:
        module = ForwardModule(obs_size, action_size)
    env = DummyEnv(spaces.Dict({"observation": observation_space, "goal": observation_space}), action_space)
    venv = make_vec_env(lambda: env)
    buffer = LGEBuffer(1_000, venv.observation_space, venv.action_space, venv, module.encoder)

    obs = venv.reset()
    for _ in range(1_000):
        action = np.array([venv.action_space.sample()])
        next_obs, reward, done, infos = venv.step(action)
        buffer.add(obs, next_obs, action, reward, done, infos)


@pytest.mark.parametrize("observation_space", OBSERVATION_SPACES)
@pytest.mark.parametrize("action_space", ACTION_SPACES)
@pytest.mark.parametrize("module_class", [AEModule, InverseModule, ForwardModule])
def test_encode(observation_space, action_space, module_class):
    obs_size = get_size(observation_space)
    action_size = get_size(action_space)
    if module_class is AEModule:
        module = AEModule(obs_size, latent_size=16)
    elif module_class is InverseModule:
        module = InverseModule(obs_size, action_size, latent_size=16)
    elif module_class is ForwardModule:
        module = ForwardModule(obs_size, action_size, latent_size=16)
    env = DummyEnv(spaces.Dict({"observation": observation_space, "goal": observation_space}), action_space)
    venv = make_vec_env(lambda: env)
    buffer = LGEBuffer(1_000, venv.observation_space, venv.action_space, venv, module.encoder)
    latent = buffer.encode(observation_space.sample())
    assert latent.shape == (16,)
    latent = buffer.encode(np.array([observation_space.sample() for _ in range(4)]))  # Batch
    assert latent.shape == (4, 16)
