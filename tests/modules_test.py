import pytest
import torch
from stable_baselines3.common.utils import set_random_seed

from lge.modules.ae_module import AEModule, CNNAEModule
from lge.modules.forward_module import CNNForwardModule, ForwardModule
from lge.modules.inverse_module import CNNInverseModule, InverseModule

BATCH_SIZE = 32
SEED = 42


@pytest.mark.parametrize("obs_size", [4, 64])
@pytest.mark.parametrize("latent_size", [4, 64])
def test_ae_module(obs_size, latent_size):
    set_random_seed(SEED)
    module = AEModule(obs_size, latent_size)
    obs = torch.randn((BATCH_SIZE, obs_size))
    output = module(obs)
    assert output.shape == (BATCH_SIZE, obs_size)
    latent = module.encoder(obs)
    assert latent.shape == (BATCH_SIZE, latent_size)


@pytest.mark.parametrize("obs_size", [4, 64])
@pytest.mark.parametrize("action_size", [4, 64])
@pytest.mark.parametrize("latent_size", [4, 64])
def test_forward_module(obs_size, action_size, latent_size):
    set_random_seed(SEED)
    module = ForwardModule(obs_size, action_size, latent_size)
    obs = torch.randn((BATCH_SIZE, obs_size))
    action = torch.randn((BATCH_SIZE, action_size))
    mean, std = module(obs, action)
    assert mean.shape == (BATCH_SIZE, obs_size)
    assert std.shape == (BATCH_SIZE, obs_size)
    latent = module.encoder(obs)
    assert latent.shape == (BATCH_SIZE, latent_size)


@pytest.mark.parametrize("obs_size", [4, 64])
@pytest.mark.parametrize("action_size", [4, 64])
@pytest.mark.parametrize("latent_size", [4, 64])
def test_inverse_module(obs_size, action_size, latent_size):
    set_random_seed(SEED)
    module = InverseModule(obs_size, action_size, latent_size)
    obs = torch.randn((BATCH_SIZE, obs_size))
    next_obs = torch.randn((BATCH_SIZE, obs_size))
    pred_action = module(obs, next_obs)
    assert pred_action.shape == (BATCH_SIZE, action_size)
    latent = module.encoder(obs)
    assert latent.shape == (BATCH_SIZE, latent_size)


@pytest.mark.parametrize("obs_shape", [(1, 84, 84), (3, 84, 84)])
@pytest.mark.parametrize("latent_size", [4, 64])
def test_cnn_ae_module(obs_shape, latent_size):
    set_random_seed(SEED)
    module = CNNAEModule(obs_shape, latent_size)
    obs = torch.randn((BATCH_SIZE, *obs_shape))
    output = module(obs)
    assert output.shape == (BATCH_SIZE, *obs_shape)
    latent = module.encoder(obs)
    assert latent.shape == (BATCH_SIZE, latent_size)


@pytest.mark.parametrize("obs_shape", [(1, 84, 84), (3, 84, 84)])
@pytest.mark.parametrize("action_size", [4, 64])
@pytest.mark.parametrize("latent_size", [4, 64])
def test_cnn_forward_module(obs_shape, action_size, latent_size):
    set_random_seed(SEED)
    module = CNNForwardModule(obs_shape, action_size, latent_size)
    obs = torch.randn((BATCH_SIZE, *obs_shape))
    action = torch.randn((BATCH_SIZE, action_size))
    mean, std = module(obs, action)
    assert mean.shape == (BATCH_SIZE, *obs_shape)
    assert std.shape == (BATCH_SIZE, *obs_shape)
    latent = module.encoder(obs)
    assert latent.shape == (BATCH_SIZE, latent_size)


@pytest.mark.parametrize("obs_shape", [(1, 84, 84), (3, 84, 84)])
@pytest.mark.parametrize("action_size", [4, 64])
@pytest.mark.parametrize("latent_size", [4, 64])
def test_cnn_inverse_module(obs_shape, action_size, latent_size):
    set_random_seed(SEED)
    module = CNNInverseModule(obs_shape, action_size, latent_size)
    obs = torch.randn((BATCH_SIZE, *obs_shape))
    next_obs = torch.randn((BATCH_SIZE, *obs_shape))
    pred_action = module(obs, next_obs)
    assert pred_action.shape == (BATCH_SIZE, action_size)
    latent = module.encoder(obs)
    assert latent.shape == (BATCH_SIZE, latent_size)
