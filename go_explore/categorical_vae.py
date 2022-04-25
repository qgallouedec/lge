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
    def __init__(self, nb_classes: int = 32, nb_categoricals: int = 32, in_channels: int = 3) -> None:
        super(CategoricalVAE, self).__init__()
        self.nb_classes = nb_classes
        self.nb_categoricals = nb_categoricals
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
            nn.Conv2d(_h * 32, _h * 64, kernel_size=3, stride=2, padding=1),  # [N x 512 x 3 x 3]
            nn.ReLU(inplace=True),
            nn.Flatten(),  # [N x 512*3*3]
            nn.Linear(_h * 64 * 3 * 3, self.nb_categoricals * self.nb_classes),  # [N x k*l]
            nn.Unflatten(-1, (self.nb_categoricals, self.nb_classes)),  # [N x k x l]
        )

        self.decoder = nn.Sequential(
            nn.Flatten(),  # [N x k*l]
            nn.Linear(self.nb_categoricals * self.nb_classes, _h * 64 * 3 * 3),  # [N x 512 x 3 x 3]
            nn.ReLU(inplace=True),
            nn.Unflatten(-1, (_h * 64, 3, 3)),
            nn.ConvTranspose2d(_h * 64, _h * 32, kernel_size=3, stride=2, padding=1),  # [N x 256 x 5 x 5]
            nn.ReLU(inplace=True),
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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        logits = self.encoder(x)
        if self.training:
            distribution = OneHotCategoricalStraightThrough(logits=logits)
            one_hot = distribution.rsample()
        else:
            argmax = torch.argmax(logits, dim=2)
            one_hot = F.one_hot(argmax, self.nb_classes).float()
        recons = self.decoder(one_hot)
        return recons, logits

    def loss_fn(self, input: torch.Tensor, recons: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the total loss.

        total loss = reconstruction loss + KL loss
        It returns the loss, and the detached reconstruction loss and KL loss for monitoring.
        """
        eps = 1e-20  # to avoid log of 0

        # Reconstruction Loss
        recons_loss = F.mse_loss(input, recons, reduction="none").sum(1)

        # KL divergence = entropy(latent) - cross_entropy(latent, uniform log-odds)
        probs = F.softmax(logits, dim=2)
        latent_entropy = probs * torch.log(probs + eps)
        target_entropy = probs * torch.log((1.0 / torch.tensor(self.nb_classes)))
        kl_divergence = torch.sum(latent_entropy - target_entropy, (1, 2))

        # total loss = reconstruction loss - KL Divergence
        loss = torch.mean(recons_loss - 0.1 * kl_divergence)
        recons_loss = torch.mean(recons_loss).item()
        kl_loss = torch.mean(-0.1 * kl_divergence).item()
        return loss, recons_loss, kl_loss
