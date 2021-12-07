import numpy as np
from stable_baselines3.common.surgeon import RewardModifier


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
        self.A = np.random.normal(size=(granularity, obs_size))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.sign(np.matmul(self.A, obs))


class SimHashMotivation(RewardModifier):
    """
    SimHash motivation.

        intrinsic_reward += β / √n(φ(obs))

    where β>0 is the bonus coefficient, φ the hash function and n the count.
    Paper: https://arxiv.org/abs/1611.04717

    :param obs_dim: observation dimension
    :type obs_dim: int
    :param granularity: granularity; higher value lead to fewer collisions
        and are thus more likely to distinguish states
    :type granularity: int
    :param beta: the bonus coefficient
    :type beta: float
    """

    def __init__(self, obs_dim: int, granularity: int, beta: float, pure_exploration: bool = False) -> None:
        self.hasher = SimHash(obs_dim, granularity)
        self.encountered_hashes = []
        self.counts = []
        self.beta = beta
        self.pure_exploration = pure_exploration

    def modify_reward(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, reward: float) -> float:
        next_obs_hash = list(self.hasher(next_obs[0]))
        if next_obs_hash not in self.encountered_hashes:  # hash is new
            self.encountered_hashes.append(next_obs_hash)
            self.counts.append(1)
        else:  # hash has already been encountered
            self.counts[self.encountered_hashes.index(next_obs_hash)] += 1
        count = self.counts[self.encountered_hashes.index(next_obs_hash)]
        intrinsic_reward = self.beta / np.sqrt(count)
        new_reward = (1 - self.pure_exploration) * reward + intrinsic_reward
        return new_reward
