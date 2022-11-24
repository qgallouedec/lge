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

SPACES = [
    spaces.Discrete(3),
    spaces.MultiDiscrete([3, 2]),
    spaces.MultiBinary(3),
    # spaces.MultiBinary([3, 2]), # Not working so far
    spaces.Box(-2, 2, shape=(2,)),
    # spaces.Box(-2, 2, shape=(2, 2)), # Not working so far
]

OBSERVATION_SPACES = [spaces.Dict({"observation": space, "goal": space}) for space in SPACES]


@pytest.mark.parametrize("observation_space", OBSERVATION_SPACES)
@pytest.mark.parametrize("action_space", SPACES)
@pytest.mark.parametrize("module_class", [AEModule, InverseModule, ForwardModule])
def test_add(observation_space, action_space, module_class):
    obs_size = get_size(observation_space["observation"])
    action_size = get_size(action_space)
    if module_class is AEModule:
        module = AEModule(obs_size)
    elif module_class is InverseModule:
        module = InverseModule(obs_size, action_size)
    elif module_class is ForwardModule:
        module = ForwardModule(obs_size, action_size)
    venv = make_vec_env(lambda: DummyEnv(observation_space, action_space))
    buffer = LGEBuffer(1_000, venv.observation_space, venv.action_space, venv, module.encoder)

    obs = venv.reset()
    for _ in range(1_000):
        action = np.array([venv.action_space.sample()])
        next_obs, reward, done, infos = venv.step(action)
        buffer.add(obs, next_obs, action, reward, done, infos)


@pytest.mark.parametrize("observation_space", OBSERVATION_SPACES)
@pytest.mark.parametrize("action_space", SPACES)
@pytest.mark.parametrize("module_class", [AEModule, InverseModule, ForwardModule])
def test_encode(observation_space, action_space, module_class):
    obs_size = get_size(observation_space["observation"])
    action_size = get_size(action_space)
    if module_class is AEModule:
        module = AEModule(obs_size, latent_size=16)
    elif module_class is InverseModule:
        module = InverseModule(obs_size, action_size, latent_size=16)
    elif module_class is ForwardModule:
        module = ForwardModule(obs_size, action_size, latent_size=16)
    venv = make_vec_env(lambda: DummyEnv(observation_space, action_space))
    buffer = LGEBuffer(1_000, venv.observation_space, venv.action_space, venv, module.encoder)
    latent = buffer.encode(venv.observation_space.sample()["observation"])
    assert latent.shape == (16,)


test_encode(OBSERVATION_SPACES[1], SPACES[0], AEModule)
