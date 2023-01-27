from typing import Tuple

import torch
from torch import Tensor, nn

from lge.modules.common import BaseModule
from lge.modules.vqvae import MultiModalVQVAE
import torch.nn.functional as F


class ForwardModule(BaseModule):
    """
    Forward module. Takes observation and action as input and predicts the next observation.

    :param obs_size: Observation size
    :param action_size: Action size
    :param latent_size: Feature size, defaults to 16
    :param net_arch: The specification of the network, default to [64, 64]
    :param activation_fn: The activation function to use for the networks, default to ReLU

                    •---------•         •---------------•
    observation --> | Encoder | ------> |               |
                    •---------•         | Forward model | --> predicted next observation
                             action --> |               |
                                        •---------------•
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        latent_size: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_size),
        )
        self.forward_model = nn.Sequential(
            nn.Linear(latent_size + action_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_size),
        )
        self.mean_net = nn.Linear(latent_size, obs_size)
        self.log_std_net = nn.Linear(latent_size, obs_size)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        """
        Return the predicted next observation given the observation and action.

        Args:
            obs (Tensor): Observation
            action (Tensor): Action

        Returns:
            Tensor: Predicted next observation
        """
        latent = self.encoder(obs)
        x = self.forward_model(torch.concat((latent, action), dim=-1))
        mean = self.mean_net(x)
        log_std = self.log_std_net(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.ones_like(mean) * log_std.exp()
        return mean, std


class _Encoder(nn.Module):
    def __init__(self, vqvae: MultiModalVQVAE, num_embeddings: int) -> None:
        super().__init__()
        self.vqvae = vqvae
        self.num_embeddings = num_embeddings

    def forward(self, input: Tensor, mod: Tensor) -> Tensor:
        codes = self.vqvae.get_codes(input, mod)
        codes = torch.reshape(F.one_hot(codes, self.num_embeddings), (input.shape[0], -1))
        return codes


class VQVAEForwardModule(nn.Module):
    """
    Vector Quantized Variational Auto-Encoder module. Takes the observation as input and predicts the observation.

    :param obs_shape: Observation shape
    :param latent_size: Feature size, defaults to 16

                    •---------•              •---------•
    observation --> | Encoder | -> latent -> | Decoder |--> predicted observation
                    •---------•              •---------•
    """

    def __init__(self, mod_size:int, num_embeddings: int = 8) -> None:
        super().__init__()
        self.vqvae = MultiModalVQVAE(embedding_dim=32, num_embeddings=num_embeddings, mod_size=mod_size)
        self.encoder = _Encoder(self.vqvae, num_embeddings)

    def forward(self, obs: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        return self.vqvae(obs, action)
