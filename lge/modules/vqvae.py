from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    """Vector Quantization (VQ) layer.

    Implements the forward pass of the VQ layer, as described in
    "Neural Discrete Representation Learning" by A. van den Oord et al. (2017)
    https://arxiv.org/abs/1711.00937
    The layer maps input latents to quantized latents, where each latent vector is replaced
    by the embedding vector that is closest to it in the euclidean sense.
    Additionally, the layer computes the vq_loss, which is the commitment loss and
    the embedding loss, as described in the paper.
    Reference: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py

    Args:
        num_embeddings (int): Number of embeddings to learn
        embedding_dim (int): Dimension of each embedding vector (D)
        beta (float): Weighting parameter for the commitment loss and embedding loss
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, a=-1 / num_embeddings, b=1 / num_embeddings)

    def get_codes(self, latents: Tensor) -> Tensor:
        """
        Computes the code for each element in `latents`.

        Args:
            latents (Tensor): Latents to be quantized. Shape is [B x D x h x w]

        Returns:
            Tensor: Codes, between 0 and num_embeddings, shape is [B x h x w]
        """
        latents = latents.permute(0, 2, 3, 1)  # [B x D x h x w] -> [B x h x w x D]
        latents_shape = latents.shape[:-1]  # (B, h, w)
        flat_latents = latents.reshape(-1, self.embedding_dim)  # [Bhw x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.norm(flat_latents[:, None] - self.embedding.weight, dim=-1)

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1)  # [Bhw]
        encoding_inds = torch.reshape(encoding_inds, latents_shape)  # [B x h x w]
        return encoding_inds

    def forward(self, latents: Tensor) -> Tuple[Tensor, Tensor]:
        latents = latents.permute(0, 2, 3, 1)  # [B x D x h x w] -> [B x h x w x D]
        latents_shape = latents.shape
        flat_latents = latents.reshape(-1, self.embedding_dim)  # [Bhw x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.norm(flat_latents[:, None] - self.embedding.weight, dim=-1)

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1)  # [Bhw,]

        # Quantize the latents
        quantized_latents = self.embedding.weight[encoding_inds]
        quantized_latents = torch.reshape(quantized_latents, latents_shape)  # [B x h x w x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss
        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents.permute(0, 3, 1, 2), vq_loss  # [B x D x h x w]


class VQVAE(nn.Module):
    def __init__(self, embedding_dim: int, num_embeddings: int, beta: float = 0.25) -> None:
        super().__init__()

        # Build Encoder
        self.encoder = nn.Sequential(  # [84 x 84 x 1]
            nn.Conv2d(1, 8, kernel_size=4, stride=2),  # [41 x 41 x 8]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),  # [19 x 19 x 16]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # [8 x 8 x 32]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # [8 x 8 x 32]
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, embedding_dim, kernel_size=1, stride=1),  # [8 x 8 x D]
            nn.LeakyReLU(inplace=True),
        )

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, beta)

        # Decoder
        self.decoder = nn.Sequential(  # [8 x 8 x D]
            nn.Conv2d(embedding_dim, 32, kernel_size=3, stride=1, padding=1),  # [8 x 8 x 32]
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, output_padding=1),  # [19 x 19 x 16]
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, output_padding=1),  # [41 x 41 x 8]
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(8, out_channels=1, kernel_size=4, stride=2),  # [84 x 84 x 1]
            nn.Tanh(),
        )

    def encode(self, input: Tensor) -> Tensor:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.

        Args:
            input (Tensor): Input tensor to encoder [B x C x H x W]

        Returns:
            Tensor: Latent codes [B x D x h x w]
        """
        return self.encoder(input)

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes onto the image space.

        Args:
            z (Tensor): Latent codes [B x D x h x w]

        Returns:
            Tensor: Predicted image [B x C x H x W]
        """
        return self.decoder(z)

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the model.

        Computes the encoding of the input, quantizes the encoding using the Vector Quantization layer,
        and passes the quantized encoding through the decoder to get the predicted output.

        Args:
            input (Tensor): The input tensor of shape [B x C x H x W]

        Returns:
            Tuple of Tensor:
                - recons (Tensor): The predicted output image [B x C x H x W]
                - vq_loss (Tensor): The Vector Quantization loss
        """
        encoding = self.encode(input)
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return self.decode(quantized_inputs), vq_loss

    def get_codes(self, input: Tensor) -> Tensor:
        """
        Encodes the input and returns the indices of the nearest latent codes in the VQ-codebook.

        Args:
            input (Tensor): Input tensor to encode [B x C x H x W]

        Returns:
            Tensor: Indices of the nearest latent codes in the VQ-codebook [B x h x w]
        """
        encoding = self.encode(input)
        return self.vq_layer.get_codes(encoding)
