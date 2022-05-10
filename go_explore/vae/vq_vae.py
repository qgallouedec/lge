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


class VQ_VAE(nn.Module):
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
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
        embedding_dim: int = 64,
        in_channels: int = 3,
    ) -> None:
        super(VQ_VAE, self).__init__()
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
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.pre_vq_conv = nn.Conv2d(256, embedding_dim, kernel_size=1, stride=1)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.post_vq_conv = nn.ConvTranspose2d(embedding_dim, 256, kernel_size=1, stride=1)

        # Decoder
        self.tconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.tconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.tconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.tconv4 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(16)
        self.tconv5 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1)
        self.bn10 = nn.BatchNorm2d(8)
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
        x = self.conv6(x)  # [N x 256 x 5 x 5]
        return x

    def decode(self, x: Tensor) -> Tensor:
        x = self.tconv1(x)  # [N x 128 x 9 x 9]
        x = self.bn6(x)
        x = F.relu(x)
        x = self.tconv2(x)  # [N x 64 x 17 x 17]
        x = self.bn7(x)
        x = F.relu(x)
        x = self.tconv3(x)  # [N x 32 x 33 x 33]
        x = self.bn8(x)
        x = F.relu(x)
        x = self.tconv4(x)  # [N x 16 x 65 x 65]
        x = self.bn9(x)
        x = F.relu(x)
        x = self.tconv5(x)  # [N x 8 x 129 x 129]
        x = self.bn10(x)
        x = F.relu(x)
        x = self.tconv6(x)  # [N x C x 129 x 129]
        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        latent = self.encode(x)
        codes = self.pre_vq_conv(latent)
        loss, quantized, perplexity, _ = self.vector_quantizer(codes)
        x = self.post_vq_conv(quantized)
        recon = self.decode(x)
        return loss, recon, perplexity
