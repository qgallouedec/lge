import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.surgeon import RewardModifier
from stable_baselines3.common.utils import get_device
from torch import nn


class Network(nn.Module):
    """Network. Used for predictor and target.

    :param obs_dim: feature dimension
    :type feature_dim: int
    :param hidden_dim: hidden dimension
    :type hidden_dim: int
    :param out_dim: output dimension
    :type out_dim: int
    """

    def __init__(self, obs_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        device = get_device("auto")
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        ).to(device)

    def forward(self, obs):
        action = self.net(obs)
        return action


class RND(RewardModifier):
    """Random Distillation Network

    :param scaling_factor: scaling factor for the intrinsic reward
    :type scaling_factor: float
    :param obs_dim: observation dimension
    :type obs_dim: int
    :param out_dim: output dimension
    :type out_dim: int
    :param hidden_dim: hidden dimension, defaults to 64
    :type hidden_dim: int, optional
    """

    def __init__(
        self,
        scaling_factor: float,
        obs_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
    ):
        self.scaling_factor = scaling_factor  # we use here a tuned scaling factor instead of normalization
        self.target = Network(obs_dim, hidden_dim, out_dim)
        self.predictor = Network(obs_dim, hidden_dim, out_dim)
        self.device = get_device("auto")

    def modify_reward(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, reward: float) -> float:
        next_obs = torch.from_numpy(next_obs).to(torch.float).to(self.device)
        with torch.no_grad():
            target = self.target(next_obs)
            pred = self.predictor(next_obs)
            intrinsic_reward = self.scaling_factor * F.mse_loss(pred, target.detach())
            new_reward = reward + self.scaling_factor * intrinsic_reward.item()
        return new_reward


class PredictorLearner(BaseCallback):
    def __init__(
        self,
        predictor: torch.nn.Module,
        target: torch.nn.Module,
        buffer: BaseBuffer,
        train_freq: int,
        grad_step: int,
        weight_decay: float,
        lr: float,
        batch_size: int,
    ) -> None:
        super(PredictorLearner, self).__init__()
        self.predictor = predictor
        self.target = target
        self.buffer = buffer
        self.train_freq = train_freq
        self.grad_step = grad_step
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = torch.nn.MSELoss()
        self.last_time_trigger = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_time_trigger) >= self.train_freq:
            self.last_time_trigger = self.num_timesteps
            self.train_once()
        return True

    def train_once(self):
        for _ in range(self.grad_step):
            try:
                batch = self.buffer.sample(self.batch_size)
            except ValueError:
                return
            pred = self.predictor(batch.observations)
            target = self.target(batch.observations)
            loss = self.criterion(pred, target.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
