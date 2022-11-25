from typing import Tuple

import torch
from torch import Tensor, nn

from lge.modules.common import BaseModule


class InverseModule(BaseModule):
    """
    Inverse module. Takes the observation and the next latent representation as input and predicts the action.

    :param obs_size: Observation size
    :param action_size: Action size
    :param latent_size: Feature size, defaults to 16

                         •---------•
         observation --> | Encoder | ---.    •---------------•
                         •---------•    '--> |               |
                                             | Inverse model | --> predicted action
                         •---------•    .--> |               |
    next observation --> | Encoder | ---'    •---------------•
                         •---------•
    """

    def __init__(self, obs_size: int, action_size: int, latent_size: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_size),
        )
        self.forward_model = nn.Sequential(
            nn.Linear(2 * latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, obs: Tensor, next_obs: Tensor) -> Tensor:
        latent = self.encoder(obs)
        next_latent = self.encoder(next_obs)
        x = torch.concat((latent, next_latent), dim=-1)
        pred_action = self.forward_model(x)
        return pred_action


class CNNInverseModule(BaseModule):
    """
    Inverse module. Takes the observation and the next latent representation as input and predicts the action.

    :param obs_size: Observation size
    :param action_size: Action size
    :param latent_size: Feature size, defaults to 16

                         •---------•
         observation --> | Encoder | ---.    •---------------•
                         •---------•    '--> |               |
                                             | Inverse model | --> predicted action
                         •---------•    .--> |               |
    next observation --> | Encoder | ---'    •---------------•
                         •---------•
    """

    def __init__(self, obs_shape: Tuple[int], action_size: int, latent_size: int = 16) -> None:
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
        self.forward_model = nn.Sequential(
            nn.Linear(2 * latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, obs: Tensor, next_obs: Tensor) -> Tensor:
        latent = self.encoder(obs)
        next_latent = self.encoder(next_obs)
        x = torch.concat((latent, next_latent), dim=-1)
        pred_action = self.forward_model(x)
        return pred_action
