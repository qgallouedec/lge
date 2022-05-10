from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CategoricalVAE(nn.Module):
    """
    Categorical Variational Auto-Encoder.

    :param nb_classes: Number of classes per categorical distribution
    :param nb_categoricals: Number of categorical distributions
    :param tau: Temparture in gumbel sampling
    :param hard_sampling: If True, the latent is sampled will be discretized as one-hot vectors
    """

    def __init__(
        self,
        nb_classes: int = 2,
        nb_categoricals: int = 16,
        tau: float = 1.0,
        hard_sampling: bool = True,
    ) -> None:
        super(CategoricalVAE, self).__init__()
        self.nb_classes = nb_classes
        self.nb_categoricals = nb_categoricals
        self.tau = tau
        self.hard_sampling = hard_sampling
        self.encoder = nn.Sequential(  # [N x C x 129 x 129]
            nn.Linear(2, 2),
            nn.ReLU(inplace=True),
            nn.Linear(2, self.nb_categoricals * self.nb_classes),  # [N x k*l]
            nn.Unflatten(-1, (self.nb_categoricals, self.nb_classes)),  # [N x k x l]
        )

        self.decoder = nn.Sequential(
            nn.Flatten(),  # [N x k*l]
            nn.Linear(self.nb_categoricals * self.nb_classes, 2),  # [N x 256 x 5 x 5]
            nn.ReLU(inplace=True),
            nn.Linear(2, 2),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self.encoder(x)
        if self.training:
            latent = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard_sampling)
        else:
            latent = F.one_hot(torch.argmax(logits, -1), self.nb_classes).float()
        recons = self.decoder(latent)
        return recons, logits


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

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        # self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.linear1 = nn.Linear(128 * 9 * 9, self.nb_categoricals * self.nb_classes)

        # Decoder
        self.linear2 = nn.Linear(self.nb_categoricals * self.nb_classes, 128 * 9 * 9)
        # self.tconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.tconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.tconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.tconv4 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(16)
        self.tconv5 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(8)
        self.tconv6 = nn.ConvTranspose2d(8, in_channels, kernel_size=3, stride=1, padding=1)

    def encode(self, x: Tensor) -> Tensor:
        # [N x C x 129 x 129]
        x = self.conv1(x)  # [N x 8 x 129 x 129]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)  # [N x 16 x 65 x 65]
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)  # [N x 32 x 33 x 33]
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)  # [N x 64 x 17 x 17]
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)  # [N x 128 x 9 x 9]
        x = self.bn5(x)
        x = F.relu(x)
        # x = self.conv6(x)  # [N x 256 x 5 x 5]
        # x = F.relu(x)
        x = x.flatten(start_dim=1)  # [N x 256*5*5]
        x = self.linear1(x)  # type: Tensor # [N x k*l]
        return x

    def decode(self, x: Tensor) -> Tensor:
        x = self.linear2(x)  # [N x 256*5*5]
        x = F.relu(x)
        x = x.unflatten(-1, (128, 9, 9))  # [N x 256 x 5 x 5]
        # x = self.tconv1(x)  # [N x 128 x 9 x 9]
        # x = F.relu(x)
        x = self.tconv2(x)  # [N x 64 x 17 x 17]
        x = self.bn6(x)
        x = F.relu(x)
        x = self.tconv3(x)  # [N x 32 x 33 x 33]
        x = self.bn7(x)
        x = F.relu(x)
        x = self.tconv4(x)  # [N x 16 x 65 x 65]
        x = self.bn8(x)
        x = F.relu(x)
        x = self.tconv5(x)  # [N x 8 x 129 x 129]
        x = self.bn9(x)
        x = F.relu(x)
        x = self.tconv6(x)  # [N x C x 129 x 129]
        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.encode(x)
        logits = x.unflatten(-1, (self.nb_categoricals, self.nb_classes))  # [N x k x l]
        if self.training:
            latent = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard_sampling)
        else:
            latent = F.one_hot(torch.argmax(logits, -1), self.nb_classes).float()
        x = latent.flatten(start_dim=1)  # [N x k*l]
        recons = self.decode(x)
        return recons, logits


class CNN_VAE(nn.Module):
    """
    Variational Auto-Encoder.

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

    def __init__(self, in_channels: int = 3) -> None:
        super(CNNCategoricalVAE, self).__init__()
        _h = 8
        self.encoder = nn.Sequential(  # [N x C x 129 x 129]
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),  # [N x 8 x 129 x 129]
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # [N x 16 x 65 x 65]
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [N x 32 x 33 x 33]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [N x 64 x 17 x 17]
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [N x 128 x 9 x 9]
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [N x 256 x 5 x 5]
            nn.ReLU(inplace=True),
            nn.Flatten(),  # [N x 256*5*5]
            nn.Linear(256 * 5 * 5, 256),  # [N x k*l]
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 256 * 5 * 5),  # [N x 256 x 5 x 5]
            nn.ReLU(inplace=True),
            nn.Unflatten(-1, (256, 5, 5)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),  # [N x 128 x 9 x 9]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),  # [N x 64 x 17 x 17]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),  # [N x 32 x 33 x 33]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),  # [N x 16 x 65 x 65]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1),  # [N x 8 x 129 x 129]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, in_channels, kernel_size=3, stride=1, padding=1),  # [N x C x 129 x 129]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self.encoder(x)
        if self.training:
            latent = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard_sampling)
        else:
            latent = F.one_hot(torch.argmax(logits, -1), self.nb_classes).float()
        recons = self.decoder(latent)
        return recons, logits
