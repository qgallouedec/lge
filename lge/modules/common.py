from typing import List, Type, Union

import torch
from stable_baselines3.common.utils import get_device
from torch import Tensor, nn


class BaseNetwork(nn.Module):
    """
    Base network.

    :param input_size: The input dimension
    :param output_size: The output dimension
    :param net_arch: The specification of the network
    :param activation_fn: The activation function to use for the networks
    :param device:
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module],
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        layers = []
        previous_layer_dim = input_size

        # Iterate through the shared layers and build the shared parts of the network
        for layer_dim in net_arch:
            layers.append(nn.Linear(previous_layer_dim, layer_dim))  # add linear of size layer
            layers.append(activation_fn())
            previous_layer_dim = layer_dim
        layers.append(nn.Linear(previous_layer_dim, output_size))

        # Create network
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Encoder(BaseNetwork):
    def __init__(
        self,
        obs_size: int,
        latent_size: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module],
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super().__init__(obs_size, latent_size, net_arch, activation_fn, device)
        self.latent_size = latent_size


class BaseModule(nn.Module):
    encoder: Encoder
