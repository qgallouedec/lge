from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, latents: Tensor) -> Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.embedding_dim)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = (
            torch.sum(flat_latents**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_latents, self.embedding.weight.t())
        )  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.num_embeddings, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]


class VQVAE(nn.Module):
    def __init__(self, embedding_dim: int, num_embeddings: int, beta: float = 0.25) -> None:
        super().__init__()

        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1),  # [84 x 84 x 1] > [42 x 42 x 8]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),  # [42 x 42 x 8] > [21 x 21 x 16]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),  #  [21 x 21 x 16] > [21 x 21 x 16]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, embedding_dim, kernel_size=1, stride=1),  #  [21 x 21 x 16] > [21 x 21 x D]
            nn.LeakyReLU(inplace=True),
        )

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, beta)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 16, kernel_size=3, stride=1, padding=1),  #  [21 x 21 x D] >  [21 x 21 x 16]
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  #  [21 x 21 x 16] > [42 x 42 x 8]
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(8, out_channels=1, kernel_size=4, stride=2, padding=1),  # [42 x 42 x 8] > [84 x 84 x 1]
            nn.Tanh(),
        )

    def encode(self, input: Tensor) -> Tensor:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.

        Args:
            input (Tensor): Input tensor to encoder [N x C x H x W]

        Returns:
            Tensor: Latent codes
        """
        return self.encoder(input)

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes onto the image space.

        Args:
            z (Tensor): Latent codes [B x D x H x W]

        Returns:
            Tensor: Predicted image [B x C x H x W]
        """
        return self.decoder(z)

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        encoding = self.encode(input)
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return self.decode(quantized_inputs), vq_loss

    def get_codes(self, input: Tensor) -> Tensor:
        encoding = self.encode(input)
        quantized_inputs, vq_loss = self.vq_layer(encoding)
