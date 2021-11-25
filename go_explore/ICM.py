from itertools import chain
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.surgeon import ActorLossModifier, RewardModifier
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import get_device
from torch import nn
from torch.nn.parameter import Parameter


class InverseModel(nn.Module):
    """Inverse model, used to predict action based on obs and next_obs.

    :param feature_dim: feature dimension
    :type feature_dim: int
    :param action_dim: action dimension
    :type action_dim: int
    :param hidden_dim: hidden dimension
    :type hidden_dim: int
    """

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        device = get_device("auto")
        self.net = nn.Sequential(
            nn.Linear(feature_dim + feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device)

    def forward(self, obs_feature, next_obs_feature):
        x = torch.concat((obs_feature, next_obs_feature), dim=-1)
        action = self.net(x)
        return action


class ForwardModel(nn.Module):
    """The forward model takes as inputs φ(st) and at and predicts the feature representation φˆ(st+1) of st+1.

    :param feature_dim: The feature dimension
    :type feature_dim: int
    :param action_dim: The action dimension
    :type action_dim: int
    :param hidden_dim: The hidden dimension
    :type hidden_dim: int
    """

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        device = get_device("auto")
        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        ).to(device)

    def forward(self, action: torch.Tensor, obs_feature: torch.Tensor) -> torch.Tensor:
        x = torch.concat((action, obs_feature), dim=-1)
        next_obs_feature = self.net(x)
        return next_obs_feature


class FeatureExtractor(nn.Module):
    """[summary]

    :param obs_dim: observation dimension
    :type obs_dim: int
    :param feature_dim: feature dimension
    :type feature_dim: int
    :param hidden_dim: hidden dimension
    :type hidden_dim: int
    """

    def __init__(self, obs_dim: int, feature_dim: int, hidden_dim: int) -> None:

        super().__init__()
        device = get_device("auto")
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        ).to(device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_feature = self.net(obs)
        return obs_feature


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
        """[summary]

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

    def modify_reward(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, reward: float) -> float:
        obs = torch.from_numpy(obs).to(torch.float).to(self.device)
        action = torch.from_numpy(action).to(torch.float).to(self.device)
        next_obs = torch.from_numpy(next_obs).to(torch.float).to(self.device)
        obs_feature = self.feature_extractor(obs)
        next_obs_feature = self.feature_extractor(next_obs)
        pred_next_obs_feature = self.forward_model(action, obs_feature)
        # Equation (6) of the original paper
        # r^i = η/2*||φˆ(st+1)−φ(st+1)||
        intrinsic_reward = self.scaling_factor * F.mse_loss(pred_next_obs_feature, next_obs_feature)
        new_reward = reward + intrinsic_reward.item()
        return new_reward
