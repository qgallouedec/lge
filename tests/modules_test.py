import pytest
import torch
from torch import nn

from lge.modules.ae_module import AEModule
from lge.modules.common import BaseNetwork, Encoder
from lge.modules.forward_module import ForwardModel, ForwardModule
from lge.modules.inverse_module import InverseModel, InverseModule


@pytest.mark.parametrize("obs_size", [4, 64])
@pytest.mark.parametrize("latent_size", [4, 64])
@pytest.mark.parametrize("net_arch", [None, [16, 16], [32, 32, 32]])
@pytest.mark.parametrize("activation_fn", [nn.ReLU, nn.Tanh])
def test_ae_module(obs_size, latent_size, net_arch, activation_fn):
    module = AEModule(obs_size, latent_size, net_arch, activation_fn)
    input = torch.randn((obs_size,))
    output = module(input)
    assert output.shape == (obs_size,)


@pytest.mark.parametrize("input_size", [4, 64])
@pytest.mark.parametrize("output_size", [4, 64])
@pytest.mark.parametrize("net_arch", [[16, 16], [32, 32, 32]])
@pytest.mark.parametrize("activation_fn", [nn.ReLU, nn.Tanh])
def test_base_network(input_size, output_size, net_arch, activation_fn):
    net = BaseNetwork(input_size, output_size, net_arch, activation_fn)
    input = torch.randn((input_size,))
    output = net(input)
    assert output.shape == (output_size,)


@pytest.mark.parametrize("obs_size", [4, 64])
@pytest.mark.parametrize("latent_size", [4, 64])
@pytest.mark.parametrize("net_arch", [[16, 16], [32, 32, 32]])
@pytest.mark.parametrize("activation_fn", [nn.ReLU, nn.Tanh])
def test_encoder(obs_size, latent_size, net_arch, activation_fn):
    net = Encoder(obs_size, latent_size, net_arch, activation_fn)
    input = torch.randn((obs_size,))
    output = net(input)
    assert output.shape == (latent_size,)


@pytest.mark.parametrize("obs_size", [4, 64])
@pytest.mark.parametrize("action_size", [4, 64])
@pytest.mark.parametrize("latent_size", [4, 64])
@pytest.mark.parametrize("net_arch", [[16, 16], [32, 32, 32]])
@pytest.mark.parametrize("activation_fn", [nn.ReLU, nn.Tanh])
def test_forward_model(obs_size, action_size, latent_size, net_arch, activation_fn):
    module = ForwardModel(obs_size, action_size, latent_size, net_arch, activation_fn)
    action, latent = torch.randn((action_size,)), torch.randn((latent_size,))
    mean, std = module(action, latent)
    assert mean.shape == (obs_size,)
    assert std.shape == (obs_size,)


@pytest.mark.parametrize("obs_size", [4, 64])
@pytest.mark.parametrize("action_size", [4, 64])
@pytest.mark.parametrize("latent_size", [4, 64])
@pytest.mark.parametrize("net_arch", [None, [16, 16], [32, 32, 32]])
@pytest.mark.parametrize("activation_fn", [nn.ReLU, nn.Tanh])
def test_forward_module(obs_size, action_size, latent_size, net_arch, activation_fn):
    module = ForwardModule(obs_size, action_size, latent_size, net_arch, activation_fn)
    obs, action = torch.randn((obs_size,)), torch.randn((action_size,))
    mean, std = module(obs, action)
    assert mean.shape == (obs_size,)
    assert std.shape == (obs_size,)


@pytest.mark.parametrize("latent_size", [4, 64])
@pytest.mark.parametrize("action_size", [4, 64])
@pytest.mark.parametrize("net_arch", [[16, 16], [32, 32, 32]])
@pytest.mark.parametrize("activation_fn", [nn.ReLU, nn.Tanh])
def test_inverse_model(latent_size, action_size, net_arch, activation_fn):
    module = InverseModel(latent_size, action_size, net_arch, activation_fn)
    latent, next_latent = torch.randn((latent_size,)), torch.randn((latent_size,))
    pred_action = module(latent, next_latent)
    assert pred_action.shape == (action_size,)


@pytest.mark.parametrize("obs_size", [4, 64])
@pytest.mark.parametrize("action_size", [4, 64])
@pytest.mark.parametrize("latent_size", [4, 64])
@pytest.mark.parametrize("net_arch", [None, [16, 16], [32, 32, 32]])
@pytest.mark.parametrize("activation_fn", [nn.ReLU, nn.Tanh])
def test_inverse_module(obs_size, action_size, latent_size, net_arch, activation_fn):
    module = InverseModule(obs_size, action_size, latent_size, net_arch, activation_fn)
    obs, next_obs = torch.randn((obs_size,)), torch.randn((obs_size,))
    pred_action = module(obs, next_obs)
    assert pred_action.shape == (action_size,)
