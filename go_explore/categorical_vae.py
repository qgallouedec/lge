from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GaussianVAE(nn.Module):
    def __init__(self, nb_classes: int = 32, nb_categoricals: int = 32, in_channels: int = 3, tau: float = 1.0) -> None:
        super(CategoricalVAE, self).__init__()
        self.nb_classes = nb_classes
        self.nb_categoricals = nb_categoricals
        self.tau = tau
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
        )
        latent_space = 512
        self.mu_net = nn.Linear(_h * 64 * 3 * 3, latent_space)
        self.log_sigma_net = nn.Linear(_h * 64 * 3 * 3, latent_space)

        self.decoder = nn.Sequential(
            nn.Linear(latent_space, _h * 64 * 3 * 3),  # [N x 512 x 3 x 3]
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
        mu = self.mu_net(logits)
        log_sigma = self.log_sigma_net(logits)
        distribution = torch.distributions.Normal(0, 1)
        z = mu + log_sigma.exp() * distribution.sample(mu.shape)
        recons = self.decoder(z)
        kl = ((2 * log_sigma).exp() + mu**2 - log_sigma - 1 / 2).mean()
        return recons, kl


class CategoricalVAE(nn.Module):
    def __init__(self, nb_classes: int = 32, nb_categoricals: int = 32, in_channels: int = 3, tau: float = 1.0) -> None:
        super(CategoricalVAE, self).__init__()
        self.nb_classes = nb_classes
        self.nb_categoricals = nb_categoricals
        self.tau = tau
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
        latent = F.gumbel_softmax(logits, tau=self.tau)
        recons = self.decoder(latent)
        # Compute kl
        probs = F.softmax(logits, dim=2)
        latent_entropy = probs * torch.log(probs + 1e-10)
        target_entropy = probs * torch.log((1.0 / torch.tensor(self.nb_classes)))
        kl = (latent_entropy - target_entropy).mean()
        return recons, kl
