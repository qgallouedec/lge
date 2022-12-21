import numpy as np
import pytest
import torch
from gym import spaces
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.utils import set_random_seed
from torch import optim

from lge.buffer import LGEBuffer
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

N_ENVS = 3
SEED = 42


@pytest.mark.parametrize("observation_space", OBSERVATION_SPACES)
@pytest.mark.parametrize("action_space", ACTION_SPACES)
@pytest.mark.parametrize("module_class", ["ae", "inverse", "forward"])
def test_add(observation_space, action_space, module_class):
    set_random_seed(SEED)

    # Create module
    action_size = get_size(action_space)
    if is_image_space(observation_space):
        obs_shape = get_shape(observation_space)
        if module_class == "ae":
            module = CNNAEModule(obs_shape, latent_size=16)
        elif module_class == "inverse":
            module = CNNInverseModule(obs_shape, action_size, latent_size=16)
        elif module_class == "forward":
            module = CNNForwardModule(obs_shape, action_size, latent_size=16)
    else:
        obs_size = get_size(observation_space)
        if module_class == "ae":
            module = AEModule(obs_size, latent_size=16)
        elif module_class == "inverse":
            module = InverseModule(obs_size, action_size, latent_size=16)
        elif module_class == "forward":
            module = ForwardModule(obs_size, action_size, latent_size=16)

    # Create environment
    def env_func():
        return DummyEnv(spaces.Dict({"observation": observation_space, "goal": observation_space}), action_space)

    venv = make_vec_env(env_func, N_ENVS)
    venv.seed(SEED)

    # Create buffer
    buffer = LGEBuffer(
        1_000,
        venv.observation_space,
        venv.action_space,
        venv,
        module.encoder,
        latent_size=16,
        n_envs=N_ENVS,
    )

    # Module to proper device
    module = module.to(buffer.device)

    # Feed the buffer
    obs = venv.reset()
    for _ in range(1_000):
        action = np.array([venv.action_space.sample() for _ in range(N_ENVS)])
        next_obs, reward, done, infos = venv.step(action)
        buffer.add(obs, next_obs, action, reward, done, infos)
        obs = next_obs


@pytest.mark.parametrize("observation_space", OBSERVATION_SPACES)
@pytest.mark.parametrize("action_space", ACTION_SPACES)
@pytest.mark.parametrize("module_class", ["ae", "inverse", "forward"])
def test_encode(observation_space, action_space, module_class):
    set_random_seed(SEED)

    # Create module
    action_size = get_size(action_space)
    if is_image_space(observation_space):
        obs_shape = get_shape(observation_space)
        if module_class == "ae":
            module = CNNAEModule(obs_shape, latent_size=16)
        elif module_class == "inverse":
            module = CNNInverseModule(obs_shape, action_size, latent_size=16)
        elif module_class == "forward":
            module = CNNForwardModule(obs_shape, action_size, latent_size=16)
    else:
        obs_size = get_size(observation_space)
        if module_class == "ae":
            module = AEModule(obs_size, latent_size=16)
        elif module_class == "inverse":
            module = InverseModule(obs_size, action_size, latent_size=16)
        elif module_class == "forward":
            module = ForwardModule(obs_size, action_size, latent_size=16)

    # Create environment
    def env_func():
        return DummyEnv(spaces.Dict({"observation": observation_space, "goal": observation_space}), action_space)

    venv = make_vec_env(env_func, N_ENVS)
    venv.seed(SEED)

    # Create buffer
    buffer = LGEBuffer(
        1_000,
        venv.observation_space,
        venv.action_space,
        venv,
        module.encoder,
        latent_size=16,
        n_envs=N_ENVS,
    )

    # Module to proper device
    module = module.to(buffer.device)

    # Test encoding
    latent = buffer.encode(observation_space.sample())
    assert latent.shape == (16,)
    latent = buffer.encode(np.array([observation_space.sample() for _ in range(4)]))  # Batch
    assert latent.shape == (4, 16)


@pytest.mark.parametrize("observation_space", OBSERVATION_SPACES)
@pytest.mark.parametrize("action_space", ACTION_SPACES)
@pytest.mark.parametrize("module_class", ["ae", "inverse", "forward"])
def test_recompute_embeddings(observation_space, action_space, module_class):
    set_random_seed(SEED)

    # Create module
    action_size = get_size(action_space)
    if is_image_space(observation_space):
        obs_shape = get_shape(observation_space)
        if module_class == "ae":
            module = CNNAEModule(obs_shape, latent_size=16)
        elif module_class == "inverse":
            module = CNNInverseModule(obs_shape, action_size, latent_size=16)
        elif module_class == "forward":
            module = CNNForwardModule(obs_shape, action_size, latent_size=16)
    else:
        obs_size = get_size(observation_space)
        if module_class == "ae":
            module = AEModule(obs_size, latent_size=16)
        elif module_class == "inverse":
            module = InverseModule(obs_size, action_size, latent_size=16)
        elif module_class == "forward":
            module = ForwardModule(obs_size, action_size, latent_size=16)

    # Create environment
    def env_func():
        return DummyEnv(spaces.Dict({"observation": observation_space, "goal": observation_space}), action_space)

    venv = make_vec_env(env_func, N_ENVS)
    venv.seed(SEED)

    # Create buffer
    buffer = LGEBuffer(
        1_000,
        venv.observation_space,
        venv.action_space,
        venv,
        module.encoder,
        latent_size=16,
        n_envs=N_ENVS,
    )

    # Module to proper device
    module = module.to(buffer.device)

    # Feed the buffer
    obs = venv.reset()
    for _ in range(500):
        action = np.array([venv.action_space.sample() for _ in range(N_ENVS)])
        next_obs, reward, done, infos = venv.step(action)
        buffer.add(obs, next_obs, action, reward, done, infos)
        obs = next_obs

    # Get and test embeddings
    buffer_emb = buffer.next_embeddings[123, 0]
    actual_emb = buffer.encode(buffer.next_observations["observation"][123, 0])
    assert np.allclose(buffer_emb, actual_emb, atol=1e-6)

    # Fake module weights update
    optimizer = optim.SGD(module.parameters(), lr=0.1)
    sample = buffer.sample(32)
    observations = preprocess(sample.observations["observation"], observation_space)
    next_observations = preprocess(sample.next_observations["observation"], observation_space)
    actions = preprocess(sample.actions, action_space)
    if module_class == "ae":
        output = module(observations)
    elif module_class == "inverse":
        output = module(observations, next_observations)
    elif module_class == "forward":
        output, _ = module(observations, actions)
    loss = torch.mean(output)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check that the current buffer_embeddings are outdated
    buffer_emb = buffer.next_embeddings[123, 0]
    actual_emb = buffer.encode(buffer.next_observations["observation"][123, 0])
    assert not np.allclose(buffer_emb, actual_emb, atol=1e-6)  # weight has been updated, not embeddings in buffer

    buffer.recompute_embeddings()

    # Check that embeddings has been updated by the method
    buffer_emb = buffer.next_embeddings[123, 0]
    actual_emb = buffer.encode(buffer.next_observations["observation"][123, 0])
    assert np.allclose(buffer_emb, actual_emb, atol=1e-6)


@pytest.mark.parametrize("observation_space", OBSERVATION_SPACES)
@pytest.mark.parametrize("action_space", ACTION_SPACES)
@pytest.mark.parametrize("module_class", ["ae", "inverse", "forward"])
def test_sample_trajectory(observation_space, action_space, module_class):
    set_random_seed(SEED)

    # Create module
    action_size = get_size(action_space)
    if is_image_space(observation_space):
        obs_shape = get_shape(observation_space)
        if module_class == "ae":
            module = CNNAEModule(obs_shape, latent_size=16)
        elif module_class == "inverse":
            module = CNNInverseModule(obs_shape, action_size, latent_size=16)
        elif module_class == "forward":
            module = CNNForwardModule(obs_shape, action_size, latent_size=16)
    else:
        obs_size = get_size(observation_space)
        if module_class == "ae":
            module = AEModule(obs_size, latent_size=16)
        elif module_class == "inverse":
            module = InverseModule(obs_size, action_size, latent_size=16)
        elif module_class == "forward":
            module = ForwardModule(obs_size, action_size, latent_size=16)

    # Create environment
    def env_func():
        return DummyEnv(spaces.Dict({"observation": observation_space, "goal": observation_space}), action_space)

    venv = make_vec_env(env_func, N_ENVS)
    venv.seed(SEED)

    # Create buffer
    buffer = LGEBuffer(
        1_000,
        venv.observation_space,
        venv.action_space,
        venv,
        module.encoder,
        latent_size=16,
        distance_threshold=0.01,
        n_envs=N_ENVS,
    )

    # Module to proper device
    module = module.to(buffer.device)

    # Feed the buffer
    obs = venv.reset()
    for _ in range(1_000):
        action = np.array([venv.action_space.sample() for _ in range(N_ENVS)])
        next_obs, reward, done, infos = venv.step(action)
        buffer.add(obs, next_obs, action, reward, done, infos)
        obs = next_obs

    # Try sampling before embedding computation
    for _ in range(10):
        subgoals, latents = buffer.sample_trajectory()  # shoudl work, even though density hasn't been computed yet

    buffer.recompute_embeddings()

    # Try sampling after embedding computation
    for _ in range(10):
        subgoals, latents = buffer.sample_trajectory()
        axis = tuple(range(1, len(subgoals.shape)))
        assert not np.any(np.all(subgoals[:-1] == subgoals[1:], axis=axis))  # consecutive goals must be different

