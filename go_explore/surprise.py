import numpy as np
import panda_gym
import torch
import torch.nn.functional
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.surgeon import RewardModifier
from stable_baselines3.common.utils import get_device
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TransitionModel(torch.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        input_size = obs_dim + action_dim
        device = get_device("auto")
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        ).to(device)
        self.mean_net = torch.nn.Linear(hidden_size, obs_dim)
        self.log_std_net = torch.nn.Linear(hidden_size, obs_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        obs_action = torch.concat((obs, action), dim=-1)
        x = self.net(obs_action)
        mean = self.mean_net(x)
        log_std = self.log_std_net(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = log_std.exp()
        normal = Normal(mean, std)
        log_prob = torch.sum(normal.log_prob(next_obs), dim=-1)
        return log_prob


class SurpriseMotivation(RewardModifier):
    def __init__(self, obs_dim: int, action_dim: int, eta: float, hidden_size: int) -> None:
        self.eta = eta
        self.transition_model = TransitionModel(obs_dim=obs_dim, action_dim=action_dim, hidden_size=hidden_size)

    def modify_reward(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, reward: float) -> float:
        obs = torch.from_numpy(obs).to(torch.float)
        action = torch.from_numpy(action).to(torch.float)
        next_obs = torch.from_numpy(next_obs).to(torch.float)

        with torch.no_grad():
            log_prob = self.transition_model(obs, action, next_obs)
        intrinsic_reward = -self.eta * log_prob.item()
        return reward + intrinsic_reward


class TransitionModelLearner(BaseCallback):
    def __init__(
        self,
        transition_model: torch.nn.Module,
        buffer: BaseBuffer,
        train_freq: int,
        grad_step: int,
        weight_decay: float,
        lr: float,
        batch_size: int,
    ) -> None:
        super(TransitionModelLearner, self).__init__()
        self.transition_model = transition_model
        self.buffer = buffer
        self.train_freq = train_freq
        self.grad_step = grad_step
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.transition_model.parameters(), lr=lr, weight_decay=weight_decay)
        self.last_time_trigger = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_time_trigger) >= self.train_freq:
            self.last_time_trigger = self.num_timesteps
            self.train_once()
        return True

    def train_once(self):
        for _ in range(self.grad_step):
            # φ_{i+1} = argmin_φ  −1/|D| sum_{(s,a,s')∈D} logPφ(s′|s,a) + α∥φ∥^2
            # D ̄KL(Pφ||Pφi)≤κ
            batch = self.buffer.sample(self.batch_size)  # (s,a,s')∈D
            log_prob = self.transition_model(batch.observations, batch.actions, batch.next_observations)
            loss = -torch.mean(log_prob)  # −1/|D| sum_{(s,a,s')∈D} logPφ(s′|s,a)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
