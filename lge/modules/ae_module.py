from typing import List, Optional, Type, Union

import torch
from torch import Tensor, nn

from lge.modules.common import BaseModule, BaseNetwork, Encoder


class AEModule(BaseModule):
    """
    Auto-encoder module. Takes the observation as input and predicts the observation.

    :param obs_size: Observation size
    :param latent_size: Feature size, defaults to 16
    :param net_arch: The specification of the network, default to [64, 64]
    :param activation_fn: The activation function to use for the networks, default to ReLU
    :param device: PyTorch device, defaults to "auto"

                    •---------•      •---------•
    observation --> | Encoder | ---> | Decoder |--> predicted observation
                    •---------•      •---------•
    """

    def __init__(
        self,
        obs_size: int,
        latent_size: int = 16,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        device: Union[torch.device, str] = "auto",
    ):
        super().__init__()
        self.obs_shape = (obs_size,)
        if net_arch is None:
            net_arch = [64, 64]

        self.encoder = Encoder(obs_size, latent_size, net_arch, activation_fn, device)
        self.decoder = BaseNetwork(latent_size, obs_size, net_arch, activation_fn, device)

    def forward(self, obs: Tensor) -> Tensor:
        latent = self.encoder(obs)
        pred_obs = self.decoder(latent)
        return pred_obs
