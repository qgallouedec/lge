import numpy as np
import pytest
from gym import spaces
from stable_baselines3 import DDPG, DQN, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.utils import set_random_seed

from lge import LatentGoExplore
from lge.buffer import LGEBuffer
from lge.lge import Goalify
from lge.modules.ae_module import AEModule, CNNAEModule
from lge.modules.forward_module import CNNForwardModule, ForwardModule
from lge.modules.inverse_module import CNNInverseModule, InverseModule
from lge.utils import get_shape, get_size
from tests.utils import BitFlippingEnv, DummyEnv

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
def test_goalify(observation_space, action_space):
    env = DummyEnv(observation_space, action_space)
    env = Goalify(env)
    assert "observation" in env.observation_space.keys()
    assert "goal" in env.observation_space.keys()
    assert env.observation_space["observation"].__class__ == env.observation_space["goal"].__class__
    print(env.action_space, env.observation_space["observation"].__class__)


@pytest.mark.parametrize("observation_space", OBSERVATION_SPACES)
@pytest.mark.parametrize("action_space", ACTION_SPACES)
def test_reset_no_buffer(observation_space, action_space):
    env = DummyEnv(observation_space, action_space)
    env = Goalify(env)
    with pytest.raises(AssertionError):
        env.reset()


@pytest.mark.parametrize("observation_space", OBSERVATION_SPACES)
@pytest.mark.parametrize("action_space", ACTION_SPACES)
@pytest.mark.parametrize("module_class", ["ae", "inverse", "forward"])
def test_goalify_reset(observation_space, action_space, module_class):
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
        env = DummyEnv(observation_space, action_space)
        env = Goalify(env)
        return env

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

    for env in venv.envs:
        env.set_buffer(buffer)

    obs = venv.reset()
    for env_idx in range(N_ENVS):
        assert observation_space.contains(obs["observation"][env_idx])
        assert observation_space.contains(obs["goal"][env_idx])


@pytest.mark.parametrize("action_type", ["discrete"])
@pytest.mark.parametrize("observation_type", ["discrete"])
@pytest.mark.parametrize("module_class", ["ae", "inverse", "forward"])
def test_goalify_step(action_type, observation_type, module_class):
    set_random_seed(SEED)

    distance_threshold = 0.01

    # Create environment
    def env_func():
        env = BitFlippingEnv(8, action_type, observation_type)
        env = Goalify(env, distance_threshold=distance_threshold)
        return env

    venv = make_vec_env(env_func, N_ENVS)
    venv.seed(SEED)

    action_space = venv.action_space
    observation_space = venv.observation_space["observation"]

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

    # Create buffer
    buffer = LGEBuffer(
        1_000,
        venv.observation_space,
        venv.action_space,
        venv,
        module.encoder,
        latent_size=16,
        distance_threshold=distance_threshold,
        n_envs=N_ENVS,
    )

    # Module to proper device
    module = module.to(buffer.device)

    for env in venv.envs:
        env.set_buffer(buffer)

    obs = venv.reset()
    env_success = [False for _ in range(N_ENVS)]
    for _ in range(8):
        actions = []
        for env_idx in range(N_ENVS):
            if action_type == "discrete":
                goal = venv.envs[env_idx].obs_to_state(obs["goal"][env_idx])
                action = np.argmax(venv.envs[env_idx].state != goal)
            actions.append(action)
        obs, reward, done, info = venv.step(actions)
        env_success = np.logical_or(env_success, reward == 0)
        for env_idx in range(N_ENVS):
            if reward[env_idx] == 0:
                assert "action_repeat" in info[env_idx].keys()
                assert info[env_idx]["action_repeat"] == actions[env_idx]
    assert np.all(env_success)  # All env must be solved


@pytest.mark.parametrize("action_type", ["discrete", "box"])
@pytest.mark.parametrize("observation_type", ["discrete", "box", "image_channel_last", "image_channel_first", "mulbinary"])
@pytest.mark.parametrize("algo", [DQN, SAC, DDPG, TD3])
@pytest.mark.parametrize("module_class", ["ae", "inverse", "forward"])
def test_lge(action_type, observation_type, algo, module_class):
    set_random_seed(SEED)
    if action_type == "discrete" and algo in [SAC, DDPG, TD3]:
        pytest.skip("Unsupported action type")
    if action_type == "box" and algo == DQN:
        pytest.skip("Unsupported action type")
    lge = LatentGoExplore(
        algo,
        "BitFlipping-v0",
        env_kwargs=dict(action_type=action_type, observation_type=observation_type),
        module_type=module_class,
        model_kwargs=dict(buffer_size=1_000),
        module_grad_steps=10,
    )
    print(lge.model.actor)
    lge.explore(total_timesteps=400)
