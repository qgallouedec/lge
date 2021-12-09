from itertools import chain
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from go_explore.icm.models import FeatureExtractor, ForwardModel, InverseModel
from stable_baselines3.common.surgeon import ActorLossModifier, RewardModifier
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import get_device
from torch.nn.parameter import Parameter


class ICM(ActorLossModifier, RewardModifier):
    def __init__(
        self,
        beta: float,
        scaling_factor: float,
        lmbda: float,
        obs_dim: int,
        action_dim: int,
        feature_dim: int = 3,
        hidden_dim: int = 64,
    ):
        """Intrinsic curiosity module.

        :param beta: scalar in [0, 1] that weighs the inverse model loss against the forward model loss
        :type beta: float
        :param scaling_factor: scaling factor for the intrinsic reward
        :type scaling_factor: float
        :param lmbda: scalar that weighs the importance of the policy gradient loss against the importance of
            learning the intrinsic reward signal
        :type lmbda: float
        :param obs_dim: observation dimension
        :type obs_dim: int
        :param action_dim: action dimension
        :type action_dim: int
        :param feature_dim: feature dimension, defaults to 3
        :type feature_dim: int, optional
        :param hidden_dim: hidden dimension, defaults to 64
        :type hidden_dim: int, optional
        """
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.lmbda = lmbda
        self.forward_model = ForwardModel(feature_dim, action_dim, hidden_dim)
        self.inverse_model = InverseModel(feature_dim, action_dim, hidden_dim)
        self.feature_extractor = FeatureExtractor(obs_dim, feature_dim, hidden_dim)
        self.device = get_device("auto")

    def parameters(self) -> Iterator[Parameter]:
        return chain(self.forward_model.parameters(), self.inverse_model.parameters(), self.feature_extractor.parameters())

    def modify_loss(self, actor_loss: torch.Tensor, replay_data: ReplayBufferSamples) -> torch.Tensor:
        obs_feature = self.feature_extractor(replay_data.observations)
        next_obs_feature = self.feature_extractor(replay_data.next_observations)
        pred_action = self.inverse_model(obs_feature, next_obs_feature)
        pred_next_obs_feature = self.forward_model(replay_data.actions, obs_feature)
        # equation (5) of the original paper
        # 1/2*||φˆ(st+1)−φ(st+1)||^2
        forward_loss = F.mse_loss(pred_next_obs_feature, next_obs_feature)
        inverse_loss = F.mse_loss(pred_action, replay_data.actions)
        # equation (7) of the original paper
        # − λEπ(st;θP )[Σtrt] + (1 − β)LI + βLF
        new_actor_loss = self.lmbda * actor_loss + (1 - self.beta) * inverse_loss + self.beta * forward_loss
        return new_actor_loss

    def modify_reward(
        self, observations: torch.Tensor, actions: torch.Tensor, next_observations: torch.Tensor, rewards: torch.Tensor
    ) -> torch.Tensor:
        obs_feature = self.feature_extractor(observations)
        next_obs_feature = self.feature_extractor(next_observations)
        pred_next_obs_feature = self.forward_model(actions, obs_feature)
        # Equation (6) of the original paper
        # r^i = η/2*||φˆ(st+1)−φ(st+1)||
        intrinsic_reward = (
            self.scaling_factor
            * torch.sum(F.mse_loss(pred_next_obs_feature, next_obs_feature, reduction="none"), dim=1).unsqueeze(1).detach()
        )
        new_reward = rewards + intrinsic_reward
        return new_reward
