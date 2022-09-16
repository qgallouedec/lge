from typing import List, Type

from torch import Tensor, nn


class BaseNetwork(nn.Module):
    """
    Base network.

    :param input_size: The input size
    :param output_size: The output size
    :param net_arch: The specification of the network
    :param activation_fn: The activation function to use for the networks
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module],
    ) -> None:
        super().__init__()
        layers = []
        previous_layer_size = input_size

        # Iterate through the shared layers and build the shared parts of the network
        for layer_size in net_arch:
            layers.append(nn.Linear(previous_layer_size, layer_size))  # add linear of size layer
            layers.append(activation_fn())
            previous_layer_size = layer_size
        layers.append(nn.Linear(previous_layer_size, output_size))

        # Create network
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Encoder(BaseNetwork):
    """
    Encoder network.

    :param obs_size: Observation size
    :param latent_size: Feature size
    :param net_arch: The specification of the network
    :param activation_fn: The activation function to use for the networks
    """

    def __init__(
        self,
        obs_size: int,
        latent_size: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module],
    ) -> None:
        super().__init__(obs_size, latent_size, net_arch, activation_fn)
        self.latent_size = latent_size


class BaseModule(nn.Module):
    encoder: Encoder  # A module is expected to contain an encoder.
