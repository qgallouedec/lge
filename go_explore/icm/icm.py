from itertools import chain
from typing import Iterator

import torch as th
import torch.nn.functional as F
from go_explore.icm.models import FeatureExtractor, ForwardModel, InverseModel
from stable_baselines3.common.surgeon import ActorLossModifier, RewardModifier
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import get_device
from torch.nn.parameter import Parameter


class ICM(ActorLossModifier, RewardModifier):
    def __init__(
        self,
        scaling_factor: float,
        actor_loss_coef: float,
        inverse_loss_coef: float,
        forward_loss_coef: float,
        obs_dim: int,
        action_dim: int,
        feature_dim: int = 16,
        hidden_dim: int = 64,
    ):
        """Intrinsic curiosity module.

        :param scaling_factor: scalar weights the intrinsic motivation
        :param actor_loss_coef: coef for the actor loss in the loss computation
        :param inverse_loss_coef: coef for the inverse loss in the loss computation
        :param forward_loss_coef: coef for the forward loss in the loss computation
        :param obs_dim: observation dimension
        :param action_dim: action dimension
        :param feature_dim: feature dimension, defaults to 16
        :param hidden_dim: hidden dimension, defaults to 64
        """
        self.scaling_factor = scaling_factor
        self.actor_loss_coef = actor_loss_coef
        self.inverse_loss_coef = inverse_loss_coef
        self.forward_loss_coef = forward_loss_coef
        self.forward_model = ForwardModel(feature_dim, action_dim, hidden_dim)
        self.inverse_model = InverseModel(feature_dim, action_dim, hidden_dim)
        self.feature_extractor = FeatureExtractor(obs_dim, feature_dim, hidden_dim)
        self.device = get_device("auto")

    def parameters(self) -> Iterator[Parameter]:
        return chain(self.forward_model.parameters(), self.inverse_model.parameters(), self.feature_extractor.parameters())

    def modify_loss(self, actor_loss: th.Tensor, replay_data: ReplayBufferSamples) -> th.Tensor:
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
        new_actor_loss = (
            self.actor_loss_coef * actor_loss + self.inverse_loss_coef * inverse_loss + self.forward_loss_coef * forward_loss
        )
        return new_actor_loss

    def modify_reward(self, replay_data: ReplayBufferSamples) -> ReplayBufferSamples:
        obs_feature = self.feature_extractor(replay_data.observations)
        next_obs_feature = self.feature_extractor(replay_data.next_observations)
        pred_next_obs_feature = self.forward_model(replay_data.actions, obs_feature)
        # Equation (6) of the original paper
        # r^i = η/2*||φˆ(st+1)−φ(st+1)||
        intrinsic_reward = (
            self.scaling_factor
            * th.sum(F.mse_loss(pred_next_obs_feature, next_obs_feature, reduction="none"), dim=1).unsqueeze(1).detach()
        )
        new_rewards = replay_data.rewards + intrinsic_reward
        new_replay_data = ReplayBufferSamples(
            replay_data.observations, replay_data.actions, replay_data.next_observations, replay_data.dones, new_rewards
        )
        return new_replay_data
