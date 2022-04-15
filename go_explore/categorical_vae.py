from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import OneHotCategoricalStraightThrough


class BetaVAE(nn.Module):
    def __init__(self, latent_size: int) -> None:
        super(BetaVAE, self).__init__()
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=6, stride=2, padding=1),  # [N x 8 x 41 x 41]
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=1),  # [N x 16 x 20 x 20]
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=6, stride=2, padding=1),  # [N x 32 x  9 x 9]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1),  # [N x 64 x  4 x 4]
            nn.ReLU(inplace=True),
            nn.Flatten(),  # [N x 64*4*4]
        )

        self.mu = nn.Linear(64 * 4 * 4, latent_size)
        self.log_var = nn.Linear(64 * 4 * 4, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64 * 4 * 4),  # [N x 64 x 4 x 4]
            nn.ReLU(inplace=True),
            nn.Unflatten(-1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1),  # [N x 32 x 9 x 9]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2, padding=1),  # [N x 16 x 20 x 20]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=1),  # [N x 8 x 41 x 41]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, kernel_size=6, stride=2, padding=1),  # [N x 1 x 84 x 84]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        latent = self.encoder(x)
        mu = self.mu(latent)
        log_var = self.log_var(latent)
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sample = mu + eps * std
        else:
            sample = mu
        recons = self.decoder(sample)
        return recons, mu, log_var


class CategoricalVAE(nn.Module):
    def __init__(self) -> None:
        super(CategoricalVAE, self).__init__()
        self.num_classes = 16
        self.nb_categoricals = 16

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=0),  # [N x 8 x 41 x 41]
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0),  # [N x 16 x 20 x 20]
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),  # [N x 32 x  9 x 9]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),  # [N x 64 x  4 x 4]
            nn.ReLU(inplace=True),
            nn.Flatten(),  # [N x 64*4*4]
            nn.Linear(64 * 4 * 4, self.nb_categoricals * self.num_classes),
            nn.Unflatten(-1, (self.nb_categoricals, self.num_classes)),
        )

        self.decoder = nn.Sequential(
            nn.Flatten(),  # [N x 32*32]
            nn.Linear(self.nb_categoricals * self.num_classes, 64 * 4 * 4),  # [N x 64 x 4 x 4]
            nn.ReLU(inplace=True),
            nn.Unflatten(-1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),  # [N x 32 x 9 x 9]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=0),  # [N x 16 x 20 x 20]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0),  # [N x 8 x 41 x 41]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=0),  # [N x 1 x 84 x 84]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        logits = self.encoder(x)
        if self.training:
            distribution = OneHotCategoricalStraightThrough(logits=logits)
            one_hot = distribution.rsample()
        else:
            argmax = torch.argmax(logits, dim=2)
            one_hot = F.one_hot(argmax, self.num_classes).float()
        recons = self.decoder(one_hot)
        recons = torch.clamp(recons, min=0.0, max=1.0)
        return recons
