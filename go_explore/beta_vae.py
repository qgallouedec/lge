from typing import Tuple

import torch
from torch import Tensor, nn


class BetaVAE(nn.Module):
    def __init__(self, latent_size: int) -> None:
        super(BetaVAE, self).__init__()
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=6, stride=2, padding=1),  # [N x 8 x 41 x 41]
            nn.ELU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=1),  # [N x 16 x 20 x 20]
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=6, stride=2, padding=1),  # [N x 32 x  9 x 9]
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1),  # [N x 64 x  4 x 4]
            nn.ELU(inplace=True),
            nn.Flatten(),  # [N x 64*4*4]
        )

        self.mu = nn.Linear(64 * 4 * 4, latent_size)
        self.log_var = nn.Linear(64 * 4 * 4, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64 * 4 * 4),  # [N x 64 x 4 x 4]
            nn.ELU(inplace=True),
            nn.Unflatten(-1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1),  # [N x 32 x 9 x 9]
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2, padding=1),  # [N x 16 x 20 x 20]
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=1),  # [N x 8 x 41 x 41]
            nn.ELU(inplace=True),
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


class CatVAE(nn.Module):
    def __init__(self, latent_size: int) -> None:
        super(CatVAE, self).__init__()
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=6, stride=2, padding=1),  # [N x 8 x 41 x 41]
            nn.ELU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=1),  # [N x 16 x 20 x 20]
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=6, stride=2, padding=1),  # [N x 32 x  9 x 9]
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1),  # [N x 64 x  4 x 4]
            nn.ELU(inplace=True),
            nn.Flatten(),  # [N x 64*4*4]
        )

        self.posterior = nn.Linear(64 * 4 * 4, 32 * 32)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64 * 4 * 4),  # [N x 64 x 4 x 4]
            nn.ELU(inplace=True),
            nn.Unflatten(-1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1),  # [N x 32 x 9 x 9]
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2, padding=1),  # [N x 16 x 20 x 20]
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=1),  # [N x 8 x 41 x 41]
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(8, 1, kernel_size=6, stride=2, padding=1),  # [N x 1 x 84 x 84]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        latent = self.encoder(x)
        x = self.posterior(latent)
        x = x.unflatten(-1, (32, 32))
        cat = torch.argmax(x, dim=1)
        one_hot = F.one_hot(cat)
        x = one_hot.flatten(-1)
        recons = self.decoder(one_hot)
        return recons, mu, log_var


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import torch.nn.functional as F
    import torchvision
    from torch import optim
    from torch.utils.data import DataLoader

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize((84, 84))])
    train_dataset = torchvision.datasets.MNIST(root="dataset", train=True, transform=transforms, download=True)
    test_dataset = torchvision.datasets.MNIST(root="dataset", train=False, transform=transforms, download=True)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)

    device = "cuda:0"
    model = CatVAE(latent_size=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    beta = 100.0
    for epoch in range(8):
        for x, _ in train_loader:
            x = x.to(device)
            recons, mu, log_var = model(x)
            recon_loss = F.mse_loss(x, recons, reduction="sum")
            kld_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)) * beta
            beta_vae_loss = recon_loss + kld_loss

            optimizer.zero_grad()
            beta_vae_loss.backward()
            optimizer.step()
        print("{:.3f}\t{:.3f}\t{:.3f}".format(beta_vae_loss.item(), recon_loss.item(), kld_loss.item()))

    def plot_ae_outputs(vae: BetaVAE, n=10):
        plt.figure(figsize=(16, 4.5))
        targets = test_dataset.targets.numpy()
        t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
            vae.eval()
            with torch.no_grad():
                rec_img, mu, log_var = vae.forward(img)
            plt.imshow(img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n // 2:
                ax.set_title("Original images")
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n // 2:
                ax.set_title("Reconstructed images")
        plt.show()

    plot_ae_outputs(model)

    import plotly.express as px
    from sklearn.manifold import TSNE
    import pandas as pd

    encoded_samples = []
    for sample, label in test_dataset:
        img = sample.unsqueeze(0).to(device)
        # Encode image
        model.eval()
        with torch.no_grad():
            encoded_img = model.encoder(img)
        # Append to list
        encoded_img = encoded_img.flatten().cpu().numpy()
        encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
        encoded_sample["label"] = label
        encoded_samples.append(encoded_sample)

    encoded_samples = pd.DataFrame(encoded_samples)
    encoded_samples

    px.scatter(encoded_samples, x="Enc. Variable 0", y="Enc. Variable 1", color=encoded_samples.label.astype(str), opacity=0.7)

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(encoded_samples.drop(["label"], axis=1))

    fig = px.scatter(
        tsne_results, x=0, y=1, color=encoded_samples.label.astype(str), labels={"0": "tsne-2d-one", "1": "tsne-2d-two"}
    )
    fig.show()
