import warnings

import torch
import torch.nn.functional as F
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import sum_independent_dims
from torch import Tensor, optim
from torch.distributions import Normal

from lge.buffer import LGEBuffer
from lge.modules.ae_module import AEModule
from lge.modules.common import BaseModule
from lge.modules.forward_module import ForwardModule
from lge.modules.inverse_module import InverseModule
from lge.utils import preprocess


class BaseLearner(BaseCallback):
    """
    Base class for learner callback.

    :param module: Module to train
    :param buffer: Buffer to sample from
    :param batch_size: Batch size, defaults to 32
    :param lr: Learning rate, defaults to 1e-3
    :param weight_decay: L2 penalty, defaults to 1e-5
    :param train_freq: Training frequency, defaults to 5_000
    :param gradient_steps: Number of gradient steps when training, defaults to 5_000
    :param first_update: Learning starts after this amount of timesteps, defaults to 5_000
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug, defaults to 0
    """

    def __init__(
        self,
        module: BaseModule,
        buffer: LGEBuffer,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        train_freq: int = 5_000,
        gradient_steps: int = 500,
        first_update: int = 5_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.module = module
        self.buffer = buffer
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.first_update = first_update

        self.optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)

    def _on_step(self):
        # Train the module every ``train_freq`` timesteps, starting at ``first_update``.
        # The latent representation is re-computed after each training phase.
        if self.n_calls == self.first_update or (self.n_calls - self.first_update) % self.train_freq == 0:
            for _ in range(self.gradient_steps):
                self.train_once()
            self.buffer.recompute_embeddings()

    def train_once(self):
        # Sample from buffer. It can fail if the buffer is not full enough.
        try:
            sample = self.buffer.sample(self.batch_size)
            observations = sample.observations
            next_observations = sample.next_observations
            actions = sample.actions
        except ValueError:
            warnings.warn(
                f"Trying to train the module when buffer before the "
                " end of the first episode. Consider increasing first_update."
            )
            return super()._on_step()

        observations = preprocess(observations, self.buffer.observation_space)
        next_observations = preprocess(next_observations, self.buffer.observation_space)
        actions = preprocess(actions, self.buffer.action_space)

        # Compute the loss
        self.module.train()
        loss = self.compute_loss(observations["observation"], next_observations["observation"], actions)

        # Step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logger.record("module/loss", loss.item())

    def compute_loss(self, observations: Tensor, next_observations: Tensor, actions: Tensor) -> Tensor:
        raise NotImplementedError()


class InverseModuleLearner(BaseLearner):
    """
    Learner for inverse module.

    :param module: Module to train
    :param buffer: Buffer to sample from
    :param batch_size: Batch size, defaults to 32
    :param lr: Learning rate, defaults to 1e-3
    :param weight_decay: L2 penalty, defaults to 1e-5
    :param train_freq: Training frequency, defaults to 5_000
    :param gradient_steps: Number of gradient steps when training, defaults to 5_000
    :param first_update: Learning starts after this amount of timesteps, defaults to 5_000
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug, defaults to 0
    """

    def __init__(
        self,
        module: InverseModule,
        buffer: LGEBuffer,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        train_freq: int = 5_000,
        gradient_steps: int = 500,
        first_update: int = 5_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(module, buffer, batch_size, lr, weight_decay, train_freq, gradient_steps, first_update, verbose)
        if type(self.buffer.action_space) == spaces.Discrete:
            self.criterion = torch.nn.CrossEntropyLoss()
        elif type(self.buffer.action_space) == spaces.Box:
            self.criterion = torch.nn.MSELoss()

    def compute_loss(self, observations: Tensor, next_observations: Tensor, actions: Tensor) -> Tensor:
        pred_actions = self.module(observations, next_observations)
        loss = self.criterion(pred_actions, actions)
        return loss


class ForwardModuleLearner(BaseLearner):
    """
    Learner for forward module.

    :param module: Module to train
    :param buffer: Buffer to sample from
    :param batch_size: Batch size, defaults to 32
    :param lr: Learning rate, defaults to 1e-3
    :param weight_decay: L2 penalty, defaults to 1e-5
    :param train_freq: Training frequency, defaults to 5_000
    :param gradient_steps: Number of gradient steps when training, defaults to 5_000
    :param first_update: Learning starts after this amount of timesteps, defaults to 5_000
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug, defaults to 0
    """

    def __init__(
        self,
        module: ForwardModule,
        buffer: LGEBuffer,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        train_freq: int = 5_000,
        gradient_steps: int = 500,
        first_update: int = 5_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(module, buffer, batch_size, lr, weight_decay, train_freq, gradient_steps, first_update, verbose)

    def compute_loss(self, observations: Tensor, next_observations: Tensor, actions: Tensor) -> Tensor:
        mean, std = self.module(observations, actions)
        distribution = Normal(mean, std)
        log_prob = distribution.log_prob(next_observations)
        log_prob = sum_independent_dims(log_prob)
        loss = -torch.mean(log_prob)  # −1/|D| sum_{(s,a,s')∈D} logPφ(s′|s,a)
        return loss


class AEModuleLearner(BaseLearner):
    """
    Learner for auto-encoder module.

    :param module: Module to train
    :param buffer: Buffer to sample from
    :param batch_size: Batch size, defaults to 32
    :param lr: Learning rate, defaults to 1e-3
    :param weight_decay: L2 penalty, defaults to 1e-5
    :param train_freq: Training frequency, defaults to 5_000
    :param gradient_steps: Number of gradient steps when training, defaults to 5_000
    :param first_update: Learning starts after this amount of timesteps, defaults to 5_000
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug, defaults to 0
    """

    def __init__(
        self,
        module: AEModule,
        buffer: LGEBuffer,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        train_freq: int = 5_000,
        gradient_steps: int = 500,
        first_update: int = 5_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(module, buffer, batch_size, lr, weight_decay, train_freq, gradient_steps, first_update, verbose)

    def compute_loss(self, observations: Tensor, next_observations: Tensor, actions: Tensor) -> Tensor:
        pred_next_observations = self.module(next_observations)  # we use next obs here
        loss = F.mse_loss(next_observations, pred_next_observations)
        return loss
