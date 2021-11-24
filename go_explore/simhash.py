from abc import ABC, abstractmethod

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper

from go_explore.wrapper import IntrinsicMotivationWrapper


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
        self.A = np.random.uniform(size=(granularity, obs_size))

    def __call__(self, obs):
        return np.sign(np.matmul(self.A, obs))



class SimHashWrapper(IntrinsicMotivationWrapper):
    """
    SimHash motivation.

        reward += β / √n(φ(obs))

    where β>0 is the bonus coefficient, φ the hash function and n the count.

    :param venv: The vectorized environment.
    :type venv: VecEnv
    :param granularity: Granularity. Higher value lead to fewer collisions
        and are thus more likely to distinguish states.
    :type granularity: int
    :param beta: The bonus coefficient.
    :type beta: float
    """

    def __init__(self, venv: VecEnv, granularity: int, beta: float) -> None:
        super().__init__(venv=venv)
        self.hasher = SimHash(venv.observation_space.shape[0], granularity)
        self.encountered_hashes = []
        self.counts = []
        self.beta = beta

    def intrinsic_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        obs_hash = list(self.hasher(obs[0]))
        if obs_hash not in self.encountered_hashes:
            self.encountered_hashes.append(obs_hash)
            self.counts.append(1)
        else:
            self.counts[self.encountered_hashes.index(obs_hash)] += 1
        intrinsic_reward = self.beta / np.sqrt(self.counts[self.encountered_hashes.index(obs_hash)])
        return intrinsic_reward
