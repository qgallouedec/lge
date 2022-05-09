from typing import Any, Dict, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from gym import Env, spaces
from stable_baselines3.common.base_class import maybe_make_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.utils import get_device
from torch import Tensor, optim
from torchvision.transforms.functional import resize

from go_explore.archive import ArchiveBuffer
from go_explore.cells import CellFactory
from go_explore.go_explore import BaseGoExplore
from go_explore.utils import is_image
from go_explore.vae import CategoricalVAE, CNNCategoricalVAE


def loss_func(input: Tensor, recons: Tensor, logits: Tensor, alpha: float = 0.01) -> Tuple[Tensor, float, float]:
    """
    TODO:

    :param input: _description_
    :param recons: _description_
    :param logits: _description_
    :param alpha:
    :return: _description_
    """
    # Reconstruction loss
    recons_loss = F.mse_loss(input, recons)

    # KL loss
    nb_classes = logits.shape[2]
    probs = F.softmax(logits, dim=2)
    latent_entropy = probs * torch.log(probs + 1e-10)
    target_entropy = probs * torch.log((1.0 / torch.tensor(nb_classes)))
    kl_loss = (latent_entropy - target_entropy).mean()

    # Total loss
    loss = recons_loss + alpha * kl_loss
    return loss, recons_loss.item(), kl_loss.item()


class VAELearner(BaseCallback):
    def __init__(
        self,
        vae: CNNCategoricalVAE,
        buffer: ArchiveBuffer,
        batch_size: int = 32,
        lr: float = 2e-4,
        train_freq: int = 1_000,
        gradient_steps: int = 1_000,
    ):
        super().__init__()
        self.vae = vae
        self.buffer = buffer
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.optimizer = optim.Adam(self.vae.parameters(), lr=lr)

    def _on_step(self):
        if self.n_calls % self.train_freq == 0:
            for _ in range(self.gradient_steps):
                self.train_once()

    def train_once(self):
        try:
            input = self.buffer.sample(self.batch_size).next_observations
        except ValueError:
            return super()._on_step()

        if type(input) is dict:
            input = input["observation"]

        if is_image(input):
            # Maps to [0, 1]
            input = resize(input, (129, 129)).float() / 255

        # Compute the output image
        self.vae.train()
        recons, logits = self.vae(input)

        # Compute the loss
        loss, recons_loss, kl_loss = loss_func(input, recons, logits)

        # Step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logger.record("vae/recons_loss", recons_loss)
        self.logger.record("vae/kl_loss", kl_loss)


class RecomputeCell(BaseCallback):
    def __init__(self, archive: ArchiveBuffer, freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.archive = archive
        self.freq = freq

    def _on_step(self):
        if self.n_calls % self.freq == 0:
            self.archive.recompute_cells()


class CategoricalVAECelling(CellFactory):
    """"""

    def __init__(self, vae: CNNCategoricalVAE) -> None:
        self.vae = vae
        self.obs_shape = (3, 210, 160)
        self.cell_space = spaces.Box(0, 1, (vae.nb_categoricals * vae.nb_classes,))

    def compute_cells(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute the cells.

        :param observations: Observations
        :return: A tensor of cells
        """
        if is_image(observations):
            input = resize(observations, (129, 129)).float() / 255
        else:
            input = observations.float()
        self.vae.eval()
        _, logits = self.vae(input)
        cell = F.one_hot(torch.argmax(logits, -1), self.vae.nb_classes)
        cell = torch.flatten(cell, start_dim=1)
        return cell


class GoExploreCatVAE(BaseGoExplore):
    """ """

    def __init__(
        self,
        model_class: Type[OffPolicyAlgorithm],
        env: Env,
        count_pow: float = 1.0,
        n_envs: int = 1,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
    ) -> None:
        env = maybe_make_env(env, verbose)
        if is_image_space(env.observation_space):
            VAE_class = CNNCategoricalVAE
        else:
            VAE_class = CategoricalVAE
        self.vae = VAE_class().to(get_device("auto"))
        cell_factory = CategoricalVAECelling(self.vae)
        super().__init__(model_class, env, cell_factory, count_pow, n_envs, replay_buffer_kwargs, model_kwargs, verbose)

    def explore(self, total_timesteps: int, update_cell_factory_freq=1_000, reset_num_timesteps: bool = False) -> None:
        """
        Run exploration.

        :param total_timesteps: Total timestep of exploration
        :param update_freq: Cells update frequency
        :param reset_num_timesteps: Whether or not to reset the current timestep number (used in logging), defaults to False
        """
        callback = [
            VAELearner(self.vae, self.archive, train_freq=update_cell_factory_freq),
            RecomputeCell(self.archive, update_cell_factory_freq),
        ]
        super().explore(total_timesteps, callback=callback, reset_num_timesteps=reset_num_timesteps)
