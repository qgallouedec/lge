import torch as th
import torch.nn.functional as F
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.surgeon import RewardModifier
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import get_device
from torch import nn


class Network(nn.Module):
    """
    Network. Used for predictor and target.

    :param obs_dim: feature dimension
    :param hidden_dim: hidden dimension
    :param out_dim: output dimension
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
    """
    Random Distillation Network.

    :param scaling_factor: scaling factor for the intrinsic reward
    :param obs_dim: observation dimension
    :param out_dim: output dimension
    :param hidden_dim: hidden dimension, defaults to 64
    """

    def __init__(
        self,
        scaling_factor: float,
        obs_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        self.scaling_factor = scaling_factor  # we use here a tuned scaling factor instead of normalization
        self.target = Network(obs_dim, hidden_dim, out_dim)
        self.predictor = Network(obs_dim, hidden_dim, out_dim)
        self.device = get_device("auto")

    def modify_reward(self, replay_data: ReplayBufferSamples) -> ReplayBufferSamples:
        target = self.target(replay_data.next_observations)
        pred = self.predictor(replay_data.next_observations)
        intrinsic_reward = self.scaling_factor * F.mse_loss(pred, target.detach(), reduction="none").mean(1).unsqueeze(1)
        new_rewards = replay_data.rewards + self.scaling_factor * intrinsic_reward.detach()
        new_replay_data = ReplayBufferSamples(
            replay_data.observations, replay_data.actions, replay_data.next_observations, replay_data.dones, new_rewards
        )
        return new_replay_data


class PredictorLearner(BaseCallback):
    def __init__(
        self,
        predictor: nn.Module,
        target: nn.Module,
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
        self.optimizer = th.optim.Adam(self.predictor.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = th.nn.MSELoss()
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
