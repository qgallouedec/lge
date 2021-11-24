from typing import Optional

import gym
import numpy as np
import panda_gym
import torch
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn
import torch.nn.functional
from go_explore.wrapper import IntrinsicMotivationWrapper


def compute_log_likelyhood(x, mean, log_std):
    """
    f(x) = prod(1/std sqrt(2*pi)) * exp(sum(-(x-mean)^2/(2*std^2)))
    log(f(x)) = sum(1/std*sqrt(2*pi)) + sum(-(x-mean)^2/(2*std^2))
    log(f(x)) = sum(1/std*sqrt(2*pi)) - sum((x-mean)^2/(2*std^2))
    log(f(x)) = sum(1/exp(log_std)*sqrt(2*pi)) - sum((x-mean)^2/(2*exp(2*log_std)))
    """
    result = torch.sum(1 / (torch.exp(log_std) * np.sqrt(2 * np.pi))) - torch.sum(
        (torch.square(x - mean)) / (2 * torch.exp(2 * log_std)), dim=-1
    )
    return result


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
        input = np.concatenate((self._last_obs, action), axis=1)
        input = torch.from_numpy(input).to(torch.float)
        obs = torch.from_numpy(obs).to(torch.float)
        with torch.no_grad():
            mean, log_std = self.transition_model(input)
        log_likelyhood = torch.nn.functional.gaussian_nll_loss(obs, mean, log_std)
        return -self.eta * log_likelyhood.item()


class TransitionModel(torch.nn.Module):
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
        self.criterion = torch.nn.GaussianNLLLoss()
        self.optimizer = torch.optim.Adam(self.transition_model.parameters(), lr=1e-4, weight_decay=1e-5)

    def _on_step(self):
        for _ in range(10):
            # φ_{i+1} = argmin_φ  −1/|D| sum_{(s,a,s')∈D} logPφ(s′|s,a) + α∥φ∥^2
            # D ̄KL(Pφ||Pφi)≤κ
            batch = self.buffer.sample(64)  # (s,a,s')∈D
            input = torch.cat((batch.observations, batch.actions), dim=1)
            mean, log_std = self.transition_model(input)
            loss = self.criterion(batch.next_observations, mean, log_std)  # −1/|D| sum_{(s,a,s')∈D} logPφ(s′|s,a)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class TransitionModelLearner(EveryNTimesteps):
    def __init__(self, transition_model: torch.nn.Module, buffer: BaseBuffer, train_freq: int, verbose: int = 0):
        callback = _TransitionModelLearner(transition_model=transition_model, buffer=buffer, verbose=verbose)
        super().__init__(n_steps=train_freq, callback=callback)
