from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Quantize(nn.Module):
    """
    Quantizer

    Args:
        embed_dim (int): Dimension of codes (quantized embedding)
        n_embed (int): Number of embeddings in the codebook
        decay (float, optional): _description_. Defaults to 0.99.
        eps (float, optional): _description_. Defaults to 1e-5.
    """

    def __init__(self, embed_dim: int, n_embed: int, decay: float = 0.99, eps: float = 1e-5) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(embed_dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Quantize the input

        Args:
            input (Tensor): Tensor of shape (..., embed_dim). Usually, (B, H, W, embed_dim)

        Returns:
            _type_: _description_
        """
        flatten = input.flatten(end_dim=-2)  # (..., embed_dim) to (K, embed_dim)
        # Compute the cross distance between every embedding and every code in the codebook.
        # Then, c_dist has shape (K, n_embed)
        dist = flatten.pow(2).sum(1, keepdim=True) - 2 * flatten @ self.embed + self.embed.pow(2).sum(0, keepdim=True)
        embed_ind = torch.argmin(dist, 1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.unflatten(dim=0, sizes=input.shape[:-1])  # unflatten : (K,) to (...,)
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))  # Get embedding from codebook; shape (..., embed_dim)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        latent_loss = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()  # Trick to propagate the gradient

        return quantize, latent_loss


class ResidualBlock(nn.Module):
    """
    Residual block

    Args:
        in_channels (int): Input channels
        residual_channels (int): Residual channels
    """

    def __init__(self, in_channels: int, residual_channels: int = 32) -> None:

        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, residual_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(residual_channels, in_channels, kernel_size=1),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.conv(input) + input


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels, stride):
        super().__init__()

        if stride == 4:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 2, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 2, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                ResidualBlock(out_channels),
                ResidualBlock(out_channels),
                nn.ReLU(inplace=True),
            )

        elif stride == 2:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 2, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1),
                ResidualBlock(out_channels),
                ResidualBlock(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, input):
        return self.net(input)


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        channel: int = 128,
    ) -> None:
        super().__init__()

        if stride == 4:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, channel, 3, padding=1),
                ResidualBlock(channel),
                ResidualBlock(channel),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, out_channels, 4, stride=2, padding=1),
            )

        elif stride == 2:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, channel, 3, padding=1),
                ResidualBlock(channel),
                ResidualBlock(channel),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel, out_channels, 4, stride=2, padding=1),
            )

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)


class VQVAE(nn.Module):
    """
    VQ-VAE

    Args:
        in_channels (int, optional): Input channels. Defaults to 3.
        channel (int, optional): _description_. Defaults to 128.
        embed_dim (int, optional): Embedding dimension. Defaults to 64.
        n_embed (int, optional): Number of embeddings in the codebook. Defaults to 512.
    """

    def __init__(self, in_channels: int = 3, channel: int = 128, embed_dim: int = 64, n_embed: int = 512) -> None:
        super().__init__()

        self.encoder_b = Encoder(in_channels, channel, stride=4)
        self.encoder_t = Encoder(channel, channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.decoder_t = Decoder(embed_dim, embed_dim, stride=2)
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
        self.decoder = Decoder(embed_dim + embed_dim, in_channels, stride=4)

    def forward(self, input: Tensor) -> Tensor:
        quantized_t, quantized_b, latent_loss = self.encode(input)
        recons = self.decode(quantized_t, quantized_b)
        return recons, latent_loss

    def encode(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        enc_b = self.encoder_b(input)
        enc_t = self.encoder_t(enc_b)

        quantized_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quantized_t, latent_loss_t = self.quantize_t(quantized_t)
        quantized_t = quantized_t.permute(0, 3, 1, 2)

        dec_t = self.decoder_t(quantized_t)
        enc_b = torch.cat([dec_t, enc_b], dim=1)

        quantized_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quantized_b, latent_loss_b = self.quantize_b(quantized_b)
        quantized_b = quantized_b.permute(0, 3, 1, 2)

        return quantized_t, quantized_b, latent_loss_t + latent_loss_b

    def decode(self, quantized_t: Tensor, quantized_b: Tensor) -> Tensor:
        upsample_t = self.upsample_t(quantized_t)
        quantized = torch.cat([upsample_t, quantized_b], 1)
        recons = self.decoder(quantized)
        return recons
