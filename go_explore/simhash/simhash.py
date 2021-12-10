from typing import Optional
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.surgeon import RewardModifier
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
import torch


class SimHash:
    """
    SimHash retrieves a binary code of observation.

        φ(obs) = sign(A*obs) ∈ {−1, 1}^granularity

    where A is a granularity × obs_size matrix with i.i.d. entries drawn from a standard
    Gaussian distribution (mean=0, std=1).

    :param obs_size: Size of observation.
    :type obs_size: int
    :param granularity: Granularity. Higher value lead to fewer collisions
        and are thus more likely to distinguish states.
    :type granularity: int
    """

    def __init__(self, obs_size: int, granularity: int) -> None:
        size = (granularity, obs_size)
        self.A = torch.normal(mean=torch.zeros(size), std=torch.ones(size))

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.sign(torch.matmul(self.A, obs.T)).T


class SimHashMotivation(RewardModifier):
    """
    SimHash motivation.

        intrinsic_reward += β / √n(φ(obs))

    where β>0 is the bonus coefficient, φ the hash function and n the count.
    Paper: https://arxiv.org/abs/1611.04717

    :param granularity: granularity; higher value lead to fewer collisions
        and are thus more likely to distinguish states
    :type granularity: int
    :param beta: the bonus coefficient
    :type beta: float
    """

    def __init__(
        self, buffer: ReplayBuffer, env: Optional[VecEnv], granularity: int, beta: float, pure_exploration: bool = False
    ) -> None:
        self.buffer = buffer
        self.env = env
        self.hasher = SimHash(buffer.obs_shape[0], granularity)
        self.beta = beta
        self.pure_exploration = pure_exploration

    def modify_reward(self, replay_data: ReplayBufferSamples) -> ReplayBufferSamples:
        next_obs_hash = self.hasher(replay_data.next_observations)
        pos = self.buffer.buffer_size if self.buffer.full else self.buffer.pos
        all_next_observations = torch.from_numpy(self.buffer._normalize_obs(self.buffer.next_observations[:pos], self.env))
        all_next_observations = all_next_observations.view(-1, self.buffer.obs_shape[0])
        all_hashes = self.hasher(all_next_observations)
        unique, all_counts = torch.unique(all_hashes, dim=0, return_counts=True)
        count = torch.zeros(next_obs_hash.shape[0])
        for k, hash in enumerate(next_obs_hash):
            idx = (unique == hash).all(1)
            count[k] = all_counts[idx]
        intrinsic_reward = self.beta / torch.sqrt(count)
        new_rewards = (1 - self.pure_exploration) * replay_data.rewards + intrinsic_reward.unsqueeze(1)
        new_replay_data = ReplayBufferSamples(
            replay_data.observations, replay_data.actions, replay_data.next_observations, replay_data.dones, new_rewards
        )
        return new_replay_data
