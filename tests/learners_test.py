import numpy as np
import pytest
from gym import spaces
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.preprocessing import is_image_space

from lge.buffer import LGEBuffer
from lge.learners import AEModuleLearner, ForwardModuleLearner, InverseModuleLearner
from lge.modules.ae_module import AEModule, CNNAEModule
from lge.modules.forward_module import CNNForwardModule, ForwardModule
from lge.modules.inverse_module import CNNInverseModule, InverseModule
from lge.utils import get_shape, get_size, preprocess
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
    # spaces.Box(-2, 2, shape=(2, 2)), # Not working, because not supported by sb3 buffer
]


@pytest.mark.parametrize("observation_space", OBSERVATION_SPACES)
@pytest.mark.parametrize("action_space", ACTION_SPACES)
@pytest.mark.parametrize("module_class", ["ae", "inverse", "forward"])
def test_learner(observation_space, action_space, module_class):
    n_envs = 1
    observation_space = spaces.Dict({"observation": observation_space, "goal": observation_space})
    latent_size = 16

    # Make the env
    def env_func():
        return DummyEnv(observation_space, action_space)

    venv = make_vec_env(env_func, n_envs)

    # Make the module
    action_size = get_size(action_space)
    if is_image_space(observation_space["observation"]):
        obs_shape = get_shape(venv.observation_space["observation"])
        if module_class == "ae":
            module = CNNAEModule(obs_shape, latent_size)
        elif module_class == "forward":
            module = CNNForwardModule(obs_shape, action_size, latent_size)
        elif module_class == "inverse":
            module = CNNInverseModule(obs_shape, action_size, latent_size)
    else:
        obs_size = get_size(venv.observation_space["observation"])
        if module_class == "ae":
            module = AEModule(obs_size, latent_size)
        elif module_class == "forward":
            module = ForwardModule(obs_size, action_size, latent_size)
        elif module_class == "inverse":
            module = InverseModule(obs_size, action_size, latent_size)

    # Make the buffer
    buffer = LGEBuffer(10_000, venv.observation_space, venv.action_space, venv, module.encoder, latent_size)

    # Make the learner
    if module_class == "ae":
        learner = AEModuleLearner(module, buffer)
    elif module_class == "forward":
        learner = ForwardModuleLearner(module, buffer)
    elif module_class == "inverse":
        learner = InverseModuleLearner(module, buffer)

    # Collect transitions
    obs = venv.reset()
    for _ in range(100):
        actions = np.array([venv.action_space.sample() for _ in range(n_envs)])
        next_obs, rewards, dones, infos = venv.step(actions)
        buffer.add(obs, next_obs, actions, rewards, dones, infos)
        obs = next_obs

    # Compute initial loss

    sample = buffer.sample(32)

    observations = preprocess(sample.observations, observation_space)["observation"]
    next_observations = preprocess(sample.next_observations, observation_space)["observation"]
    actions = preprocess(sample.actions, action_space)

    # Compute the initial loss
    module.eval()
    initial_loss = learner.compute_loss(observations, next_observations, actions)

    # Train
    for _ in range(10):
        learner.train_once()

    # Compute the final loss
    module.eval()
    final_loss = learner.compute_loss(observations, next_observations, actions)
    if module_class in ["inverse", "forward"]:  # TODO: Env purely stochastic, action/obs unpredictable
        assert final_loss < initial_loss
