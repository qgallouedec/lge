from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class VectorQuantizer(nn.Module):
    """
    From an encoding, find the closest embedding from the code book

    :param num_embeddings: Number of embeddings in the codebook
    :param embedding_dim: Size of the embedding
    :param commitment_cost: Weight of the commitment in the loss computation
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: int) -> None:
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.codebook.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1)
        input_shape = inputs.shape

        # Flatten input (B x H x W x C) to (B*H*W x C)
        flat_input = inputs.reshape(-1, self.embedding_dim)

        # Calculate distances: result size: B*H*W x num_embeddings
        distances = torch.cdist(flat_input, self.codebook.weight)

        # Encoding
        encodings_idxs = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encodings_idxs, num_classes=self.num_embeddings).float()

        # Quantize and unflatten
        quantized = self.codebook(encodings_idxs).view(input_shape)
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2), perplexity, encodings


class Residual(nn.Module):
    def __init__(self, n_channels):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        residual = x
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + residual


class ResidualStack(nn.Module):
    def __init__(self, n_channels):
        super(ResidualStack, self).__init__()
        self.residual1 = Residual(n_channels)
        self.residual2 = Residual(n_channels)

    def forward(self, x):
        x = self.residual1(x)
        x = F.relu(x)
        x = self.residual2(x)
        x = F.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=4, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        # self.conv_3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.residual_stack = ResidualStack(n_channels=out_channels)

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        # x = F.relu(x)
        # x = self.conv_3(x)
        return self.residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
        self.residual_stack = ResidualStack(n_channels=num_hiddens)
        self.tconv1 = nn.ConvTranspose2d(
            in_channels=num_hiddens, out_channels=num_hiddens // 2, kernel_size=4, stride=2, padding=1
        )
        self.tconv2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2, out_channels=3, kernel_size=4, stride=2, padding=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.residual_stack(x)
        x = self.tconv1(x)
        x = F.relu(x)
        x = self.tconv2(x)
        return x


class VQ_VAE(nn.Module):
    def __init__(self, num_hiddens, num_embeddings, embedding_dim, commitment_cost):
        super(VQ_VAE, self).__init__()

        self.encoder = Encoder(3, num_hiddens)
        self.pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)
        self.vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, num_hiddens)

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        loss, quantized, perplexity, _ = self.vq_vae(z)
        x_recon = self.decoder(quantized)

        return loss, x_recon, perplexity
