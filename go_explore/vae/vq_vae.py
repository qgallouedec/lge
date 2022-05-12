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

    def get_quantized(self, inputs: Tensor) -> Tensor:
        # Calculate distances: result size: B*H*W x num_embeddings
        distances = torch.cdist(inputs, self.codebook.weight)

        # Encoding
        encodings_idxs = torch.argmin(distances, dim=1)

        # Quantize and unflatten
        quantized = self.codebook(encodings_idxs)
        return quantized

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Calculate distances: result size: B*H*W x num_embeddings
        distances = torch.cdist(inputs, self.codebook.weight)

        # Encoding
        encodings_idxs = torch.argmin(distances, dim=1)

        # Quantize and unflatten
        quantized = self.codebook(encodings_idxs)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        # Compute perplexity
        encodings = F.one_hot(encodings_idxs, num_classes=self.num_embeddings).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return vq_loss, quantized, perplexity


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
        num_embeddings: int = 8,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
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
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)  # [N x 16 x 65 x 65]
        x = F.relu(x)
        x = self.bn2(x)
        x = self.conv3(x)  # [N x 32 x 33 x 33]
        x = F.relu(x)
        x = self.bn3(x)
        x = self.conv4(x)  # [N x 64 x 17 x 17]
        x = F.relu(x)
        x = self.bn4(x)
        x = self.conv5(x)  # [N x 128 x 9 x 9]
        x = F.relu(x)
        x = self.bn5(x)
        x = self.conv6(x)  # [N x 256 x 5 x 5]
        return x

    def decode(self, x: Tensor) -> Tensor:
        x = self.tconv1(x)  # [N x 128 x 9 x 9]
        x = F.relu(x)
        x = self.bn6(x)
        x = self.tconv2(x)  # [N x 64 x 17 x 17]
        x = F.relu(x)
        x = self.bn7(x)
        x = self.tconv3(x)  # [N x 32 x 33 x 33]
        x = F.relu(x)
        x = self.bn8(x)
        x = self.tconv4(x)  # [N x 16 x 65 x 65]
        x = F.relu(x)
        x = self.bn9(x)
        x = self.tconv5(x)  # [N x 8 x 129 x 129]
        x = F.relu(x)
        x = self.bn10(x)
        x = self.tconv6(x)  # [N x C x 129 x 129]
        return x

    def get_quantized(self, x: Tensor) -> Tensor:
        latent = self.encode(x)
        codes = self.pre_vq_conv(latent)
        # convert inputs from BCHW -> BHWC
        codes = codes.permute(0, 2, 3, 1)
        codes_shape = codes.shape
        # Flatten input (B x H x W x C) to (B*H*W x C)
        # codes = codes.reshape(-1, self.embedding_dim)
        codes = codes.flatten(start_dim=0, end_dim=2)
        quantized = self.vector_quantizer.get_quantized(codes)
        quantized = quantized.view(codes_shape).permute(0, 3, 1, 2)
        return quantized

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        latent = self.encode(x)
        codes = self.pre_vq_conv(latent)
        # convert inputs from BCHW -> BHWC
        codes = codes.permute(0, 2, 3, 1)
        codes_shape = codes.shape

        # Flatten input (B x H x W x C) to (B*H*W x C)
        codes = codes.flatten(start_dim=0, end_dim=2)
        vq_loss, quantized, perplexity = self.vector_quantizer(codes)
        quantized = quantized.view(codes_shape).permute(0, 3, 1, 2)

        x = self.post_vq_conv(quantized)
        recons = self.decode(x)
        return recons, vq_loss, perplexity


class Linear_VQ_VAE(nn.Module):
    """
    Linear VQ Variational Auto-Encoder.

    :param nb_classes: Number of classes per categorical distribution
    :param nb_categoricals: Number of categorical distributions
    :param in_channels: Number of input channels
    :param tau: Temparture in gumbel sampling
    :param hard_sampling: If True, the latent is sampled will be discretized as one-hot vectors
    """

    def __init__(
        self,
        in_features: int,
        num_embeddings: int = 10,
        embedding_dim: int = 2,
        commitment_cost: float = 0.25,
    ) -> None:
        super(Linear_VQ_VAE, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Encoder
        self.fc1 = nn.Linear(in_features, in_features)
        self.fc2 = nn.Linear(in_features, in_features)

        self.vector_quantizer = VectorQuantizer(self.num_embeddings, self.embedding_dim, commitment_cost)

        # Decoder
        self.fc3 = nn.Linear(in_features, in_features)
        self.fc4 = nn.Linear(in_features, in_features)

    def encode(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def decode(self, x: Tensor) -> Tensor:
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x

    def get_quantized(self, x: Tensor) -> Tensor:
        latent = self.encode(x)
        batch_size = latent.shape[0]
        # (B x L*C) to (B x L x C)
        codes = latent.view(batch_size, -1, self.embedding_dim)
        # (B x L x C) to (B*L x C)
        codes = codes.flatten(start_dim=0, end_dim=1)

        _, quantized, _ = self.vector_quantizer(codes)
        return quantized

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        latent = self.encode(x)
        batch_size = latent.shape[0]
        # (B x L*C) to (B x L x C)
        codes = latent.view(batch_size, -1, self.embedding_dim)
        # (B x L x C) to (B*L x C)
        codes = codes.flatten(start_dim=0, end_dim=1)

        vq_loss, quantized, perplexity = self.vector_quantizer(codes)

        # (B*L x C) to (B x L x C)
        quantized = quantized.view(batch_size, -1, self.embedding_dim)
        # (B x L x C) to (B x L*C)
        quantized = quantized.flatten(start_dim=1, end_dim=2)

        recons = self.decode(quantized)
        return recons, vq_loss, perplexity


# class VQ_VAE(nn.Module):
#     """Vector Quantized AutoEncoder for mnist"""

#     def __init__(self, hidden=200, num_embeddings=10, embedding_dim=12, vq_coef=0.2, comit_coef=0.4):
#         super(VQ_VAE, self).__init__()

#         self.embedding_dim = embedding_dim
#         self.num_embeddings = num_embeddings

#         self.fc1 = nn.Linear(784, 400)
#         self.fc2 = nn.Linear(400, self.embedding_dim * self.num_embeddings)

#         self.emb = NearestEmbed(num_embeddings, self.embedding_dim)

#         self.fc3 = nn.Linear(self.embedding_dim * self.num_embeddings, 400)
#         self.fc4 = nn.Linear(400, 784)

#         self.vq_coef = vq_coef
#         self.comit_coef = comit_coef
#         self.hidden = hidden
#         self.ce_loss = 0
#         self.vq_loss = 0
#         self.commit_loss = 0

#     def encode(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = x.view(-1, self.num_embeddings, self.embedding_dim, int(self.hidden / self.embedding_dim))
#         return x

#     def decode(self, z):
#         z = self.fc3(z)
#         z = F.relu(z)
#         z = self.fc4(z)
#         return z

#     def forward(self, x):
#         z_e = self.encode(x.view(-1, 784))
#         z_q, _ = self.emb(z_e, weight_sg=True).view(-1, self.hidden)
#         emb, _ = self.emb(z_e.detach()).view(-1, self.hidden)
#         return self.decode(z_q), z_e, emb
