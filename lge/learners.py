from typing import Callable

import torch
from stable_baselines3.common.callbacks import BaseCallback
from torch import Tensor, optim
from torch.distributions import Normal

from lge.archive import ArchiveBuffer
from lge.modules.ae_module import AEModule
from lge.modules.common import BaseModule
from lge.modules.forward_module import ForwardModule
from lge.modules.inverse_module import InverseModule
from lge.utils import is_image


def sum_independent_dims(tensor: Tensor) -> Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.
    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class BaseLearner(BaseCallback):
    def __init__(
        self,
        module: BaseModule,
        archive: ArchiveBuffer,
        batch_size: int = 32,
        criterion: Callable = torch.nn.MSELoss(),
        lr: float = 1e-3,
        train_freq: int = 10_000,
        gradient_steps: int = 10_000,
        first_update: int = 3_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.module = module
        self.archive = archive
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.first_update = first_update

        self.criterion = criterion
        self.optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=1e-5)

    def _on_step(self):
        if self.n_calls == self.first_update or (self.n_calls - self.first_update) % self.train_freq == 0:
            for _ in range(self.gradient_steps):
                self.train_once()
            self.archive.recompute_embeddings()

    def train_once(self):
        try:
            sample = self.archive.sample(self.batch_size)
            observations = sample.observations
            next_observations = sample.next_observations
            actions = sample.actions
        except ValueError:
            return super()._on_step()

        if type(observations) is dict:
            observations = observations["observation"]
            next_observations = next_observations["observation"]

        # Convert all to float
        observations = observations.float()
        next_observations = next_observations.float()

        # Squeeze needed when cross entropy loss
        actions = sample.actions.squeeze()

        if is_image(observations):
            observations = observations / 255
            next_observations = next_observations / 255

        # Compute the loss
        self.module.train()
        loss = self.compute_loss(observations, next_observations, actions)

        # Step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logger.record("module/loss", loss.item())

    def compute_loss(self, observations: Tensor, next_observations: Tensor, actions: Tensor) -> Tensor:
        raise NotImplementedError()


class InverseModuleLearner(BaseLearner):
    def __init__(
        self,
        module: InverseModule,
        archive: ArchiveBuffer,
        batch_size: int = 32,
        criterion: Callable = torch.nn.MSELoss(),
        lr: float = 0.001,
        train_freq: int = 10000,
        gradient_steps: int = 10000,
        first_update: int = 3000,
        verbose: int = 0,
    ) -> None:
        super().__init__(module, archive, batch_size, criterion, lr, train_freq, gradient_steps, first_update, verbose)

    def compute_loss(self, observations: Tensor, next_observations: Tensor, actions: Tensor) -> Tensor:
        pred_actions = self.module(observations, next_observations)
        loss = self.criterion(pred_actions, actions)
        return loss


class ForwardModuleLearner(BaseLearner):
    def __init__(
        self,
        module: ForwardModule,
        archive: ArchiveBuffer,
        batch_size: int = 32,
        criterion: Callable = torch.nn.MSELoss(),
        lr: float = 0.001,
        train_freq: int = 10000,
        gradient_steps: int = 10000,
        first_update: int = 3000,
        verbose: int = 0,
    ) -> None:
        super().__init__(module, archive, batch_size, criterion, lr, train_freq, gradient_steps, first_update, verbose)

    def compute_loss(self, observations: Tensor, next_observations: Tensor, actions: Tensor) -> Tensor:
        mean, std = self.module(observations, actions)
        distribution = Normal(mean, std)
        log_prob = distribution.log_prob(next_observations)
        log_prob = sum_independent_dims(log_prob)
        loss = -torch.mean(log_prob)  # −1/|D| sum_{(s,a,s')∈D} logPφ(s′|s,a) + α∥φ∥^2
        return loss


class AEModuleLearner(BaseLearner):
    def __init__(
        self,
        module: AEModule,
        archive: ArchiveBuffer,
        batch_size: int = 32,
        criterion: Callable = torch.nn.MSELoss(),
        lr: float = 0.001,
        train_freq: int = 10000,
        gradient_steps: int = 10000,
        first_update: int = 3000,
        verbose: int = 0,
    ) -> None:
        super().__init__(module, archive, batch_size, criterion, lr, train_freq, gradient_steps, first_update, verbose)

    def compute_loss(self, observations: Tensor, next_observations: Tensor, actions: Tensor) -> Tensor:
        pred_next_observations = self.module(next_observations)  # we use next obs here
        loss = self.criterion(pred_next_observations, next_observations)
        return loss
