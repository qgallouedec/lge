from typing import Tuple

import torch
from torch import Tensor, nn

from lge.modules.common import BaseModule


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
            nn.ReLU(),
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


class CNNAEModule(BaseModule):
    """
    CNN Auto-encoder module. Takes the observation as input and predicts the observation.

    :param obs_shape: Observation shape
    :param latent_size: Feature size, defaults to 16

                    •---------•              •---------•
    observation --> | Encoder | -> latent -> | Decoder |--> predicted observation
                    •---------•              •---------•
    """

    def __init__(self, obs_shape: Tuple[int], latent_size: int = 16) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            _shape = self.cnn(torch.zeros((1, *obs_shape))).shape[1:]
            n_flatten = _shape.numel()

        self.encoder = nn.Sequential(
            self.cnn,
            nn.Flatten(),
            nn.Linear(n_flatten, latent_size),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, n_flatten),
            nn.ReLU(),
            nn.Unflatten(1, _shape),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, obs_shape[0], kernel_size=8, stride=4, padding=0),
        )

    def forward(self, obs: Tensor) -> Tensor:
        latent = self.encoder(obs)
        pred_obs = self.decoder(latent)
        return pred_obs
