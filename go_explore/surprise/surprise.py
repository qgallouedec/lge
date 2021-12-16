import torch as th
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.surgeon import RewardModifier
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import get_device
from torch import nn, optim
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class TransitionModel(nn.Module):
    def __init__(self, obs_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        input_size = obs_size + action_size
        device = get_device("auto")
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        ).to(device)
        self.mean_net = nn.Linear(hidden_size, obs_size).to(device)
        self.log_std_net = nn.Linear(hidden_size, obs_size).to(device)

    def forward(self, obs: th.Tensor, action: th.Tensor, next_obs: th.Tensor) -> th.Tensor:
        obs_action = th.concat((obs, action), dim=-1)
        x = self.net(obs_action)
        mean = self.mean_net(x)
        log_std = self.log_std_net(x)
        log_std = th.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = th.ones_like(mean) * log_std.exp()
        distribution = Normal(mean, std)
        log_prob = distribution.log_prob(next_obs)
        log_prob = sum_independent_dims(log_prob)
        return log_prob


class SurpriseMotivation(RewardModifier):
    def __init__(self, obs_size: int, action_size: int, eta_0: float, hidden_size: int) -> None:
        self.eta_0 = eta_0
        self.transition_model = TransitionModel(obs_size, action_size, hidden_size)
        self.device = get_device("auto")

    def modify_reward(self, replay_data: ReplayBufferSamples) -> ReplayBufferSamples:
        log_prob = self.transition_model(replay_data.observations, replay_data.actions, replay_data.next_observations)
        eta = self.eta_0 / max(1, th.mean(-log_prob))
        intrinsic_rewards = -eta * log_prob
        new_rewards = replay_data.rewards + intrinsic_rewards.unsqueeze(1)
        new_replay_data = ReplayBufferSamples(
            replay_data.observations, replay_data.actions, replay_data.next_observations, replay_data.dones, new_rewards
        )
        return new_replay_data


class TransitionModelLearner(BaseCallback):
    def __init__(
        self,
        transition_model: nn.Module,
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
        self.optimizer = optim.Adam(self.transition_model.parameters(), lr=lr, weight_decay=weight_decay)
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
            try:
                batch = self.buffer.sample(self.batch_size)  # (s,a,s')∈D
            except ValueError:
                return
            log_prob = self.transition_model(batch.observations, batch.actions, batch.next_observations)
            loss = -th.mean(log_prob)  # −1/|D| sum_{(s,a,s')∈D} logPφ(s′|s,a)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
