from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class InverseModel(nn.Module):
    encoder: nn.Module
    latent_inverse_model: nn.Module
    latent_size: int
    obs_shape: Tuple

    def forward(self, obs: Tensor, next_obs: Tensor) -> Tensor:
        latent = self.encoder(obs)
        next_latent = self.encoder(next_obs)
        x = torch.concat((latent, next_latent), dim=-1)
        pred_action = self.latent_inverse_model(x)
        return pred_action


class LinearInverseModel(InverseModel):
    """
    Linear Inverse Model. Predict the action from the observation and the next observation.
    The same encoder is used for both observation and next_observation.

    Encoder is composed of fc -> relu -> fc -> batch norm.
    Inverse model is composed of fc -> relu -> fc.

    :param latent_size: Size of the output of the encoder
    :param width: width of the network

                 •---------•
         obs --> | Encoder | ---.    •---------------•
                 •---------•    '--> |               |
                                     | Inverse model | --> predicted action
                 •---------•    .--> |               |
    next_obs --> | Encoder | ---'    •---------------•
                 •---------•
    """

    def __init__(self, obs_size: int, action_size: int, latent_size: int, width: int = 16) -> None:
        super(LinearInverseModel, self).__init__()
        self.latent_size = latent_size
        self.obs_shape = (obs_size,)
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, width),
            nn.ReLU(),
            nn.Linear(width, latent_size),
        )
        # Inverse latent model
        self.latent_inverse_model = nn.Sequential(
            nn.Linear(2 * latent_size, width),
            nn.ReLU(),
            nn.Linear(width, action_size),
        )


class ConvInverseModel(InverseModel):
    """
    Linear Inverse Model. Predict the action from the observation and the next observation.
    The same encoder is used for both observation and next_observation.


                 •---------•
         obs --> | Encoder | ---.    •---------------•
                 •---------•    '--> |               |
                                     | Inverse model | --> predicted action
                 •---------•    .--> |               |
    next_obs --> | Encoder | ---'    •---------------•
                 •---------•

    Encoder:

    | Size      | Channels       |
    |-----------|----------------|
    | 129 x 129 | 3              |
    | 129 x 129 | 8              |
    | 65 x 65   | 16             |
    | 33 x 33   | 32             |
    | 17 x 17   | 64             |
    | 9 x 9     | 128            |
    | 5 x 5     | 256            |

    The result is flattened to a vector of size 6400.
    The result is passed through a fully connected layer that utput size is nb_categoricals x nb_classes vector.
    The latent is sampled from the Gumbel-Softmax distribution.
    The result is passed through a fully connected layer that utput size is 6400.
    The result is unflattened to a vector of size 3 x 3 x 512.

    Decoder:

    | Size      | Channels       |
    |-----------|----------------|
    | 5 x 5     | 256            |
    | 9 x 9     | 128            |
    | 17 x 17   | 64             |
    | 33 x 33   | 32             |
    | 65 x 65   | 16             |
    | 129 x 129 | 8              |
    | 129 x 129 | input channels |

    :param nb_classes: Number of classes per categorical distribution
    :param nb_categoricals: Number of categorical distributions
    :param in_channels: Number of input channels
    :param tau: Temparture in gumbel sampling
    :param hard_sampling: If True, the latent is sampled will be discretized as one-hot vectors
    """

    def __init__(self, action_size: int, latent_size: int) -> None:
        super(ConvInverseModel, self).__init__()
        self.latent_size = latent_size
        self.obs_shape = (12, 84, 84)
        self.encoder = nn.Sequential(  # [N x 12 x 84 x 84]
            nn.Conv2d(12, 64, kernel_size=3, stride=1),  # [64 x 82 x 82]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=3),  # [64 x 27 x 27]
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # [128 x 25 x 25]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=3),  # [128 x 8 x 8]
            nn.Conv2d(128, 128, kernel_size=3, stride=1),  # [128 x 6 x 6]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [128 x 3 x 3]
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, latent_size),
        )
        # Inverse latent model
        self.latent_inverse_model = nn.Sequential(
            nn.Linear(2 * latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )
