from torch import nn
import torch
from stable_baselines3.common.utils import get_device


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
    """The forward model takes as inputs φ(st) and at and predicts the feature representation φˆ(st+1).

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
    """Feature extractor.

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
