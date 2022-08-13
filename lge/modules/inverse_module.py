from typing import List, Optional, Type, Union

import torch
from torch import Tensor, nn

from lge.modules.common import BaseModule, BaseNetwork, Encoder


class InverseModel(BaseNetwork):
    """
    Forward model. Takes the latent representation and the next latent representation as input and predicts the action.

    :param latent_size: Feature size
    :param action_size: Action size
    :param net_arch: The specification of the network
    :param activation_fn: The activation function to use for the networks
    :param device: PyTorch device, defaults to "auto"
    """

    def __init__(
        self,
        latent_size: int,
        action_size: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module],
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super().__init__(2 * latent_size, action_size, net_arch, activation_fn, device)

    def forward(self, latent: Tensor, next_latent: Tensor) -> Tensor:
        x = torch.concat((latent, next_latent), dim=-1)
        pred_action = self.net(x)
        return pred_action


class InverseModule(BaseModule):
    """
    Inverse module. Takes the observation and the next latent representation as input and predicts the action.

    :param obs_size: Observation size
    :param action_size: Action size
    :param latent_size: Feature size, defaults to 16
    :param net_arch: The specification of the network, default to [64, 64]
    :param activation_fn: The activation function to use for the networks, default to ReLU
    :param device: PyTorch device, defaults to "auto"

                         •---------•
         observation --> | Encoder | ---.    •---------------•
                         •---------•    '--> |               |
                                             | Inverse model | --> predicted action
                         •---------•    .--> |               |
    next observation --> | Encoder | ---'    •---------------•
                         •---------•
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        latent_size: int = 16,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super().__init__()
        if net_arch is None:
            net_arch = [64, 64]

        self.encoder = Encoder(obs_size, latent_size, net_arch, activation_fn, device)
        self.forward_model = InverseModel(latent_size, action_size, net_arch, activation_fn, device)

    def forward(self, obs: Tensor, next_obs: Tensor) -> Tensor:
        latent = self.encoder(obs)
        next_latent = self.encoder(next_obs)
        pred_action = self.forward_model(latent, next_latent)
        return pred_action
