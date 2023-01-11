import warnings

import torch
import torch.nn.functional as F
from gym import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import sum_independent_dims
from torch import Tensor, optim
from torch.distributions import Normal

from lge.buffer import LGEBuffer
from lge.modules.ae_module import AEModule, VQVAEModule
from lge.modules.common import BaseModule
from lge.modules.forward_module import ForwardModule
from lge.modules.inverse_module import InverseModule
from lge.utils import preprocess


class BaseLearner(BaseCallback):
    """
    Base class for module learner. A learner is a callback used for learning a module.

    Args:
        module (BaseModule): Module to train
        buffer (LGEBuffer): Buffer to sample from
        batch_size (int, optional): Batch size. Defaults to 64.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): L2 penalty. Defaults to 1e-5.
        train_freq (int, optional): Training frequency. Defaults to 5_000.
        gradient_steps (int, optional): Number of gradient steps when training. Defaults to 500.
        learning_starts (int, optional): Learning starts after this amount of timesteps. Defaults to 100.
        verbose (int, optional): The verbosity level: 0 none, 1 training information, 2 debug. Defaults to 0.
    """

    def __init__(
        self,
        module: BaseModule,
        buffer: LGEBuffer,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        train_freq: int = 5_000,
        gradient_steps: int = 500,
        learning_starts: int = 100,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.module = module
        self.buffer = buffer
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.learning_starts = learning_starts

        self.optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)

    def _on_step(self):
        # Train the module every ``train_freq`` timesteps, starting at ``learning_starts``.
        # The latent representation is re-computed after each training phase.
        if self.num_timesteps >= self.learning_starts:
            if (self.num_timesteps - self.learning_starts) // self.model.n_envs % (self.train_freq // self.model.n_envs) == 0:
                self.logger.log("Training module...")
                for _ in range(self.gradient_steps):
                    self.train_once()
                self.logger.log("Recomputing embeddings...")
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
                "end of the first episode. Consider increasing learning_starts."
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

        if self.logger is not None:
            self.logger.record("module/loss", loss.item())

    def compute_loss(self, observations: Tensor, next_observations: Tensor, actions: Tensor) -> Tensor:
        raise NotImplementedError()


class InverseModuleLearner(BaseLearner):
    """
    Learner for inverse module.

    Args:
        module (InverseModule): Inverse module to train
        buffer (LGEBuffer): Buffer to sample from
        batch_size (int, optional): Batch size. Defaults to 64.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): L2 penalty. Defaults to 1e-5.
        train_freq (int, optional): Training frequency. Defaults to 5_000.
        gradient_steps (int, optional): Number of gradient steps when training. Defaults to 500.
        learning_starts (int, optional): Learning starts after this amount of timesteps. Defaults to 100.
        verbose (int, optional): The verbosity level: 0 none, 1 training information, 2 debug. Defaults to 0.
    """

    def __init__(
        self,
        module: InverseModule,
        buffer: LGEBuffer,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        train_freq: int = 5_000,
        gradient_steps: int = 500,
        learning_starts: int = 100,
        verbose: int = 0,
    ) -> None:
        super().__init__(module, buffer, batch_size, lr, weight_decay, train_freq, gradient_steps, learning_starts, verbose)
        if type(self.buffer.action_space) in [spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary]:
            self.criterion = torch.nn.CrossEntropyLoss()
        elif type(self.buffer.action_space) == spaces.Box:
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError(f"Action space {self.buffer.action_space} not supported.")

    def compute_loss(self, observations: Tensor, next_observations: Tensor, actions: Tensor) -> Tensor:
        pred_actions = self.module(observations, next_observations)
        loss = self.criterion(pred_actions, actions)
        return loss


class ForwardModuleLearner(BaseLearner):
    """
    Learner for forward module.

    Args:
        module (ForwardModule): Forward module to train
        buffer (LGEBuffer): Buffer to sample from
        batch_size (int, optional): Batch size. Defaults to 64.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): L2 penalty. Defaults to 1e-5.
        train_freq (int, optional): Training frequency. Defaults to 5_000.
        gradient_steps (int, optional): Number of gradient steps when training. Defaults to 500.
        learning_starts (int, optional): Learning starts after this amount of timesteps. Defaults to 100.
        verbose (int, optional): The verbosity level: 0 none, 1 training information, 2 debug. Defaults to 0.
    """

    def __init__(
        self,
        module: ForwardModule,
        buffer: LGEBuffer,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        train_freq: int = 5_000,
        gradient_steps: int = 500,
        learning_starts: int = 100,
        verbose: int = 0,
    ) -> None:
        super().__init__(module, buffer, batch_size, lr, weight_decay, train_freq, gradient_steps, learning_starts, verbose)

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

    Args:
        module (InverseModule): Auto-encoder module to train
        buffer (LGEBuffer): Buffer to sample from
        batch_size (int, optional): Batch size. Defaults to 64.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): L2 penalty. Defaults to 1e-5.
        train_freq (int, optional): Training frequency. Defaults to 5_000.
        gradient_steps (int, optional): Number of gradient steps when training. Defaults to 500.
        learning_starts (int, optional): Learning starts after this amount of timesteps. Defaults to 100.
        verbose (int, optional): The verbosity level: 0 none, 1 training information, 2 debug. Defaults to 0.
    """

    def __init__(
        self,
        module: AEModule,
        buffer: LGEBuffer,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        train_freq: int = 5_000,
        gradient_steps: int = 500,
        learning_starts: int = 100,
        verbose: int = 0,
    ) -> None:
        super().__init__(module, buffer, batch_size, lr, weight_decay, train_freq, gradient_steps, learning_starts, verbose)

    def compute_loss(self, observations: Tensor, next_observations: Tensor, actions: Tensor) -> Tensor:
        pred_next_observations = self.module(next_observations)  # we use next obs here
        loss = F.mse_loss(next_observations, pred_next_observations)
        return loss


class VQVAEModuleLearner(BaseLearner):
    """
    Learner for VQ variational auto-encoder module.

    Args:
        module (VAEModule): Auto-encoder module to train
        buffer (LGEBuffer): Buffer to sample from
        latent_loss_coef (float): Coefficient for the latent loss
        batch_size (int, optional): Batch size. Defaults to 64.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (float, optional): L2 penalty. Defaults to 1e-5.
        train_freq (int, optional): Training frequency. Defaults to 5_000.
        gradient_steps (int, optional): Number of gradient steps when training. Defaults to 500.
        learning_starts (int, optional): Learning starts after this amount of timesteps. Defaults to 100.
        verbose (int, optional): The verbosity level: 0 none, 1 training information, 2 debug. Defaults to 0.
    """

    def __init__(
        self,
        module: VQVAEModule,
        buffer: LGEBuffer,
        latent_loss_coef: float = 0.25,
        batch_size: int = 64,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        train_freq: int = 5_000,
        gradient_steps: int = 500,
        learning_starts: int = 100,
        verbose: int = 0,
    ) -> None:
        super().__init__(module, buffer, batch_size, lr, weight_decay, train_freq, gradient_steps, learning_starts, verbose)
        self.latent_loss_coef = latent_loss_coef

    def compute_loss(self, observations: Tensor, next_observations: Tensor, actions: Tensor) -> Tensor:
        pred_next_observations, latent_loss = self.module(next_observations)
        recons_loss = F.mse_loss(next_observations, pred_next_observations)
        loss = recons_loss + self.latent_loss_coef * latent_loss
        return loss
