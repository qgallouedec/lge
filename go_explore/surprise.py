from typing import Optional

import gym
import numpy as np
import panda_gym
import torch
import torch.nn.functional
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn
from torch import nn
from torch.distributions import Normal

from go_explore.wrapper import IntrinsicMotivationWrapper


class SurpriseWrapper(IntrinsicMotivationWrapper):
    def __init__(
        self,
        venv: VecEnv,
        transition_model: torch.nn.Module,
        eta: float,
        observation_space: Optional[gym.spaces.Space] = None,
        action_space: Optional[gym.spaces.Space] = None,
    ):
        super().__init__(venv, observation_space=observation_space, action_space=action_space)
        self.eta = eta
        self.transition_model = transition_model
        self.criterion = torch.nn.MSELoss()

    def reset(self) -> VecEnvObs:
        self._last_obs = self.venv.reset()
        return self._last_obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = super().step_wait()
        self._last_obs = obs
        return obs, reward, done, info

    def intrinsic_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        next_obs = torch.from_numpy(obs).to(torch.float)
        obs = torch.from_numpy(self._last_obs).to(torch.float)
        action = torch.from_numpy(action).to(torch.float)
        
        with torch.no_grad():
            log_prob = self.transition_model(obs, action, next_obs)

        return -self.eta * log_prob.item()


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TransitionModel(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_shape + action_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_net = nn.Linear(hidden_dim, obs_shape)
        self.log_std_net = nn.Linear(hidden_dim, obs_shape)

    def forward(self, obs, action, next_obs):
        x = torch.concat((obs, action), dim=-1)
        x = self.net(x)
        mean = self.mean_net(x)
        log_std = self.log_std_net(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = log_std.exp()
        normal = Normal(mean, std)
        log_prob = torch.sum(normal.log_prob(next_obs), dim=-1)
        return log_prob



class _TransitionModel(torch.nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()
        input_size = env.observation_space.shape[0] + env.action_space.shape[0]
        output_size = env.observation_space.shape[0] + env.observation_space.shape[0]  # mean and std
        self.obs_size = env.observation_space.shape[0]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
        )
        self.mean_net = torch.nn.Linear(64, env.observation_space.shape[0])
        self.std_net = torch.nn.Sequential(
            torch.nn.Linear(64, env.observation_space.shape[0]),
            torch.nn.Softmax(1),
        )

    def forward(self, x):
        y = self.net(x)
        mean = self.mean_net(y)
        log_std = self.std_net(y)
        return mean, log_std


class _TransitionModelLearner(BaseCallback):
    def __init__(self, transition_model: torch.nn.Module, buffer: BaseBuffer, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.buffer = buffer
        self.transition_model = transition_model
        self.optimizer = torch.optim.Adam(self.transition_model.parameters(), lr=1e-4, weight_decay=1e-5)

    def _on_step(self):
        for _ in range(10):
            # φ_{i+1} = argmin_φ  −1/|D| sum_{(s,a,s')∈D} logPφ(s′|s,a) + α∥φ∥^2
            # D ̄KL(Pφ||Pφi)≤κ
            batch = self.buffer.sample(64)  # (s,a,s')∈D
            log_prob = self.transition_model(batch.observations, batch.actions, batch.next_observations)
            loss = -torch.mean(log_prob)  # −1/|D| sum_{(s,a,s')∈D} logPφ(s′|s,a)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class TransitionModelLearner(EveryNTimesteps):
    def __init__(self, transition_model: torch.nn.Module, buffer: BaseBuffer, train_freq: int, verbose: int = 0):
        callback = _TransitionModelLearner(transition_model=transition_model, buffer=buffer, verbose=verbose)
        super().__init__(n_steps=train_freq, callback=callback)
