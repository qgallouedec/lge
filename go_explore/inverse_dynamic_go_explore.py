from typing import Any, Callable, Dict, Optional, Type

import torch
import torch.nn.functional as F
from gym import Env, spaces
from stable_baselines3.common.base_class import maybe_make_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.utils import get_device
from torch import optim

from go_explore.archive import ArchiveBuffer
from go_explore.go_explore import BaseGoExplore
from go_explore.inverse_model import ConvInverseModel, LinearInverseModel
from go_explore.utils import ImageSaver, is_image


class InverseModelLearner(BaseCallback):
    def __init__(
        self,
        archive: ArchiveBuffer,
        batch_size: int = 128,
        criterion: Callable = torch.nn.MSELoss(),
        lr: float = 1e-3,
        train_freq: int = 10_000,
        gradient_steps: int = 10_000,
        first_update: int = 3_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.archive = archive
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.first_update = first_update

        self.criterion = criterion
        self.optimizer = optim.Adam(self.archive.inverse_model.parameters(), lr=lr, weight_decay=1e-5)

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

        # Compute the output image
        self.archive.inverse_model.train()
        pred_actions = self.archive.inverse_model(observations, next_observations)

        # Compute the loss
        loss = self.criterion(pred_actions, actions)

        # Step the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logger.record("inverse_model/pred_loss", loss.item())


class GoExploreInverseModel(BaseGoExplore):
    """ """

    def __init__(
        self,
        model_class: Type[OffPolicyAlgorithm],
        env: Env,
        density_pow: float = -2.0,
        traj_step: int = 3,
        distance_threshold: float = 1.0,
        n_envs: int = 1,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
    ) -> None:
        env = maybe_make_env(env, verbose)
        if is_image_space(env.observation_space):
            inverse_model = ConvInverseModel(env.action_space.n, 16).to(get_device("auto"))
        else:
            inverse_model = LinearInverseModel(
                obs_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], latent_size=2
            ).to(get_device("auto"))
        super().__init__(
            model_class,
            env,
            inverse_model,
            density_pow,
            traj_step,
            distance_threshold,
            n_envs,
            replay_buffer_kwargs,
            model_kwargs,
            verbose,
        )

    def explore(self, total_timesteps: int, update_cell_factory_freq=1_000, reset_num_timesteps: bool = False) -> None:
        """
        Run exploration.

        :param total_timesteps: Total timestep of exploration
        :param update_freq: Cells update frequency
        :param reset_num_timesteps: Whether or not to reset the current timestep number (used in logging), defaults to False
        """
        if type(self.model.env.action_space) == spaces.Discrete:
            criterion = torch.nn.CrossEntropyLoss()
        elif type(self.model.env.action_space) == spaces.Box:
            criterion = torch.nn.MSELoss()
        callback = [
            InverseModelLearner(
                self.archive,
                criterion=criterion,
                train_freq=update_cell_factory_freq,
                gradient_steps=update_cell_factory_freq,
                first_update=1_000,
            ),
            ImageSaver(self.model.env, save_freq=5_000),
        ]
        super().explore(total_timesteps, callback=callback, reset_num_timesteps=reset_num_timesteps)
