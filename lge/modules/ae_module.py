from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lge.modules.common import BaseModule
from lge.vqvae import VQVAE


class AEModule(BaseModule):
    """
    Auto-encoder module. Takes the observation as input and predicts the observation.

    :param obs_size: Observation size
    :param latent_size: Feature size, defaults to 16

                    •---------•              •---------•
    observation --> | Encoder | -> latent -> | Decoder |--> predicted observation
                    •---------•              •---------•
    """

    def __init__(self, obs_size: int, latent_size: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, obs_size),
        )

    def forward(self, obs: Tensor) -> Tensor:
        latent = self.encoder(obs)
        pred_obs = self.decoder(latent)
        return pred_obs


class _Encoder(nn.Module):
    def __init__(self, vqvae: VQVAE, num_embeddings: int) -> None:
        super().__init__()
        self.vqvae = vqvae
        self.num_embeddings = num_embeddings

    def forward(self, input: Tensor) -> Tensor:
        codes = self.vqvae.get_codes(input)
        codes = torch.reshape(F.one_hot(codes, self.num_embeddings), (input.shape[0], -1))
        return codes


class VQVAEModule(nn.Module):
    """
    Vector Quantized Variational Auto-Encoder module. Takes the observation as input and predicts the observation.

    :param obs_shape: Observation shape
    :param latent_size: Feature size, defaults to 16

                    •---------•              •---------•
    observation --> | Encoder | -> latent -> | Decoder |--> predicted observation
                    •---------•              •---------•
    """

    def __init__(self, num_embeddings: int = 8) -> None:
        super().__init__()
        self.vqvae = VQVAE(embedding_dim=32, num_embeddings=num_embeddings)
        self.encoder = _Encoder(self.vqvae, num_embeddings)

    def forward(self, obs: Tensor):
        return self.vqvae(obs)
