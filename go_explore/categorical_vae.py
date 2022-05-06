from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CNNCategoricalVAE(nn.Module):
    """
    Categorical Variational Auto-Encoder.

    Encoder:

    | Size      | Channels       |
    |-----------|----------------|
    | 129 x 129 | input channels |
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

    def __init__(
        self,
        nb_classes: int = 32,
        nb_categoricals: int = 32,
        in_channels: int = 3,
        tau: float = 1.0,
        hard_sampling: bool = True,
    ) -> None:
        super(CNNCategoricalVAE, self).__init__()
        self.nb_classes = nb_classes
        self.nb_categoricals = nb_categoricals
        self.tau = tau
        self.hard_sampling = hard_sampling
        _h = 8
        self.encoder = nn.Sequential(  # [N x C x 129 x 129]
            nn.Conv2d(in_channels, _h * 1, kernel_size=3, stride=1, padding=1),  # [N x 8 x 129 x 129]
            nn.ReLU(inplace=True),
            nn.Conv2d(_h * 1, _h * 2, kernel_size=3, stride=2, padding=1),  # [N x 16 x 65 x 65]
            nn.ReLU(inplace=True),
            nn.Conv2d(_h * 2, _h * 4, kernel_size=3, stride=2, padding=1),  # [N x 32 x 33 x 33]
            nn.ReLU(inplace=True),
            nn.Conv2d(_h * 4, _h * 8, kernel_size=3, stride=2, padding=1),  # [N x 64 x 17 x 17]
            nn.ReLU(inplace=True),
            nn.Conv2d(_h * 8, _h * 16, kernel_size=3, stride=2, padding=1),  # [N x 128 x 9 x 9]
            nn.ReLU(inplace=True),
            nn.Conv2d(_h * 16, _h * 32, kernel_size=3, stride=2, padding=1),  # [N x 256 x 5 x 5]
            nn.ReLU(inplace=True),
            nn.Flatten(),  # [N x 256*5*5]
            nn.Linear(_h * 32 * 5 * 5, self.nb_categoricals * self.nb_classes),  # [N x k*l]
            nn.Unflatten(-1, (self.nb_categoricals, self.nb_classes)),  # [N x k x l]
        )

        self.decoder = nn.Sequential(
            nn.Flatten(),  # [N x k*l]
            nn.Linear(self.nb_categoricals * self.nb_classes, _h * 32 * 5 * 5),  # [N x 256 x 5 x 5]
            nn.ReLU(inplace=True),
            nn.Unflatten(-1, (_h * 32, 5, 5)),
            nn.ConvTranspose2d(_h * 32, _h * 16, kernel_size=3, stride=2, padding=1),  # [N x 128 x 9 x 9]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(_h * 16, _h * 8, kernel_size=3, stride=2, padding=1),  # [N x 64 x 17 x 17]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(_h * 8, _h * 4, kernel_size=3, stride=2, padding=1),  # [N x 32 x 33 x 33]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(_h * 4, _h * 2, kernel_size=3, stride=2, padding=1),  # [N x 16 x 65 x 65]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(_h * 2, _h * 1, kernel_size=3, stride=2, padding=1),  # [N x 8 x 129 x 129]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(_h * 1, in_channels, kernel_size=3, stride=1, padding=1),  # [N x C x 129 x 129]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self.encoder(x)
        if self.training:
            latent = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard_sampling)
        else:
            latent = F.one_hot(torch.argmax(logits, -1), self.nb_classes).float()
        recons = self.decoder(latent)
        return recons, logits
