from typing import Tuple

import torch
from torch import Tensor, nn

from lge.modules.common import BaseModule


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


class CNNForwardModule(BaseModule):
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
        )

        self.decoder_mean = nn.Sequential(
            nn.Linear(latent_size + action_size, n_flatten),
            nn.ReLU(),
            nn.Unflatten(1, _shape),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, obs_shape[0], kernel_size=8, stride=4, padding=0),
        )
        self.decoder_std = nn.Sequential(
            nn.Linear(latent_size + action_size, n_flatten),
            nn.ReLU(),
            nn.Unflatten(1, _shape),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, obs_shape[0], kernel_size=8, stride=4, padding=0),
        )

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        """
        Return the predicted next observation given the observation and action.

        Args:
            obs (Tensor): Observation
            action (Tensor): Action

        Returns:
            Tensor: Predicted next observation and standard deviation
        """
        latent = self.encoder(obs)
        x = torch.concat((latent, action), dim=-1)
        mean = self.decoder_mean(x)
        log_std = self.decoder_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.ones_like(mean) * log_std.exp()
        return mean, std
