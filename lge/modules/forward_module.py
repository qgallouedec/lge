from typing import List, Optional, Type, Union

import torch
from torch import Tensor, nn

from lge.modules.common import BaseModule, BaseNetwork, Encoder


class ForwardModel(BaseNetwork):
    """
    Forward model. Predict the next latent representation based on observation and action.

    :param latent_size: Feature size
    :param net_arch: The specification of the network
    :param action_size: Action size
    :param activation_fn: The activation function to use for the networks
    :param device:
    """

    def __init__(
        self,
        latent_size: int,
        action_size: int,
        obs_size: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module],
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super().__init__(latent_size + action_size, obs_size, net_arch, activation_fn, device)

    def forward(self, latent: Tensor, action: Tensor) -> Tensor:
        x = torch.concat((latent, action), dim=-1)
        pred_obs = self.net(x)
        return pred_obs


class ForwardModule(BaseModule):
    """
    Forward module. From the observation and the action, predicts the next feature representation.

    :param obs_size: Observation dimension
    :param action_size: Action size
    :param latent_size: Feature size, defaults to 16
    :param net_arch: The specification of the network, default to [64, 64]
    :param activation_fn: The activation function to use for the networks, default to ReLU
    :param device:


            •---------•         •---------------•
    obs --> | Encoder | ------> |               |
            •---------•         | Forward model | --> predicted next observation
                        action --> |               |
                                •---------------•
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        latent_size: int = 16,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        device: Union[torch.device, str] = "auto",
    ):

        super().__init__()
        if net_arch is None:
            net_arch = [64, 64]

        self.encoder = Encoder(obs_size, latent_size, net_arch, activation_fn, device)
        self.forward_model = ForwardModel(obs_size, action_size, latent_size, net_arch, activation_fn, device)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        latent = self.encoder(obs)
        pred_obs = self.forward_model(latent, action)
        return pred_obs
