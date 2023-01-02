from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from gym import spaces
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from torch import Tensor, nn

from lge.utils import batchify, estimate_density, is_batched, lighten, preprocess, sample_geometric_with_max


class LGEBuffer(HerReplayBuffer):
    """
    LGE buffer.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param env: The training environment
    :param encoder: Encoder to compute latent representation
    :param latent_size: Feature size, defaults to 16
    :param distance_threshold: The goal is reached when the latent distance between
        the current obs and the goal obs is below this threshold, defaults to 1.0
    :param p: Geometric parameter for final goal sampling, defaults to 0.005
    :param device: PyTorch device, defaults to "auto"
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Unused, defaults to False
    :param handle_timeout_termination: Unsed, defaults to True
    :param n_sampled_goal: Number of virtual transitions to create per real transition,
        by sampling new goals, defaults to 4
    :param goal_selection_strategy: Strategy for sampling goals for replay.
        One of ['episode', 'final', 'future'], defaults to "future"
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        env: VecEnv,
        encoder: nn.Module,
        latent_size: int,
        distance_threshold: float = 1.0,
        p: float = 0.005,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        n_sampled_goal: int = 4,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
    ) -> None:
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            env,
            device=get_device(device),
            n_envs=n_envs,
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy=goal_selection_strategy,
        )

        self.distance_threshold = distance_threshold
        self.encoder = encoder
        self.latent_size = latent_size
        self.p = p

        self.goal_embeddings = np.zeros((self.buffer_size, self.n_envs, self.latent_size), dtype=np.float32)
        self.next_embeddings = np.zeros((self.buffer_size, self.n_envs, self.latent_size), dtype=np.float32)

        # The buffer does not compute density after every new transition stored. The densities are
        # computed when the method recompute_embeddings() is appealed. To keep track of the number
        # densities computed, we use self.self.nb_density_computed
        self.nb_density_computed = 0

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        is_virtual: bool = False,
    ) -> None:
        pos = self.pos  # Store before incrementation
        super().add(obs, next_obs, action, reward, done, infos, is_virtual)
        for env_idx in range(self.n_envs):
            if done[env_idx]:
                episode_start = self.ep_start[pos, env_idx]
                episode_end = self.pos
                if episode_end < episode_start:
                    # Occurs when the buffer becomes full, the storage resumes at the
                    # beginning of the buffer. This can happen in the middle of an episode.
                    episode_end += self.buffer_size
                episode = np.arange(episode_start, episode_end) % self.buffer_size

                self.goal_embeddings[episode, env_idx] = self.encode(self.observations["goal"][episode, env_idx])
                self.next_embeddings[episode, env_idx] = self.encode(self.next_observations["observation"][episode, env_idx])

    def recompute_embeddings(self) -> None:
        """
        Re-compute all the embeddings and estimate the density. This method must
        be called on a regular basis to keep the density estimation up to date.
        """
        upper_bound = self.pos if not self.full else self.buffer_size

        for env_idx in range(self.n_envs):
            # Recompute 256 by 256 to avoid cuda space allocation error.
            self.goal_embeddings[:upper_bound, env_idx] = self.encode(self.observations["goal"][:upper_bound, env_idx])
            self.next_embeddings[:upper_bound, env_idx] = self.encode(
                self.next_observations["observation"][:upper_bound, env_idx]
            )

        self.nb_density_computed = upper_bound * self.n_envs

        # Reshape and convert embeddings to torch tensor
        all_embeddings = self.next_embeddings[:upper_bound]
        all_embeddings = all_embeddings.reshape(self.nb_density_computed, -1)
        all_embeddings = self.to_torch(all_embeddings)

        # Estimate density based on the embeddings
        density = np.zeros((self.nb_density_computed), dtype=np.float32)
        k = 0
        while k < self.nb_density_computed:
            upper = min(self.nb_density_computed, k + 256)
            embeddings = all_embeddings[k:upper]
            density[k:upper] = estimate_density(embeddings, all_embeddings).detach().cpu().numpy()
            k += 256
        self.density = density
        self.sorted_density = np.argsort(density)

    def encode(self, obs: Union[int, np.ndarray]) -> np.ndarray:
        """
        Encode an observation.

        :param obs: The observation to encode
        :return: The latent representation
        """
        _is_batched = is_batched(obs, self.observation_space["observation"])  # whether obs is batched
        if not _is_batched:
            obs = batchify(obs)

        batch_size = obs.shape[0]

        step = 2048
        k = 0
        embeddings = np.zeros((batch_size, self.latent_size), dtype=np.float32)
        while k < batch_size:
            upper = min(batch_size, k + step)
            _obs = self.to_torch(obs[k:upper])
            self.encoder.eval()
            _obs = preprocess(_obs, self.observation_space["observation"])
            embeddings[k:upper] = self.encoder(_obs).detach().cpu().numpy()
            k += step
        if _is_batched:
            return embeddings
        else:
            return embeddings.squeeze(0)

    def sample_trajectory(self, lighten_dist_coef: float = 1.0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Sample a trajcetory of observations based on the latent density.

        :param lighten_dist_coef: Remove subgoal that are not further than lighten_dist_coef*dist_threshold
            from the previous subgoal, defaults to 1.0
        :return: The list of subgoals and their latent representation
        """
        if self.nb_density_computed == 0:  # no density computed yet
            goal = np.expand_dims(self.observation_space["goal"].sample(), 0)
            return goal, self.encode(goal)

        goal_id = self.sorted_density[sample_geometric_with_max(self.p, max_value=self.sorted_density.shape[0]) - 1]
        goal_pos = goal_id // self.n_envs
        goal_env = goal_id % self.n_envs
        episode_start = self.ep_start[goal_pos, goal_env]
        episode_end = goal_pos + 1
        # When the buffer is full, the return-to-start mechanism may cause a sampled trajectory to be
        # partially stored at the beginning and end of the buffer. Handle this situation appropriately.
        episode_end += self.buffer_size * (episode_end < episode_start)
        episode = np.arange(episode_start, episode_end) % self.buffer_size
        trajectory = self.next_observations["observation"][episode, goal_env]
        emb_trajectory = self.next_embeddings[episode, goal_env]
        # Lighten trajectory
        idxs = lighten(emb_trajectory, self.distance_threshold * lighten_dist_coef)
        trajectory, emb_trajectory = trajectory[idxs], emb_trajectory[idxs]
        return trajectory, emb_trajectory

    def _get_real_samples(
        self, batch_inds: np.ndarray, env_indices: np.ndarray, env: Optional[VecNormalize] = None
    ) -> DictReplayBufferSamples:
        """
        Get the samples corresponding to the batch and environment indices.

        :param batch_inds: Indices of the transitions
        :param env_indices: Indices of the envrionments
        :param env: associated gym VecEnv to normalize the
            observations/rewards when sampling, defaults to None
        :return: Samples
        """
        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices] for key, obs in self.observations.items()})
        next_obs_ = self._normalize_obs({key: obs[batch_inds, env_indices] for key, obs in self.next_observations.items()})
        next_embeddings = self.next_embeddings[batch_inds, env_indices]
        goal_embeddings = self.goal_embeddings[batch_inds, env_indices]

        # Compute new reward
        dist = np.linalg.norm(goal_embeddings - next_embeddings, axis=1)
        is_success = dist < self.distance_threshold
        rewards = is_success.astype(np.float32) - 1
        for idx, info in enumerate(self.infos[batch_inds, env_indices]):
            if info.get("dead", False):
                rewards[idx] -= 100

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])),
            rewards=self.to_torch(self._normalize_reward(rewards, env)),
        )

    def _get_virtual_samples(
        self, batch_inds: np.ndarray, env_indices: np.ndarray, env: Optional[VecNormalize] = None
    ) -> DictReplayBufferSamples:
        """
        Get the samples corresponding to the batch and environment indices.

        :param batch_inds: Indices of the transitions
        :param env_indices: Indices of the envrionments
        :param env: associated gym VecEnv to normalize the observations/rewards when sampling, defaults to None
        :return: Samples
        """
        # Get infos and obs
        obs = {key: obs[batch_inds, env_indices] for key, obs in self.observations.items()}
        next_obs = {key: obs[batch_inds, env_indices] for key, obs in self.next_observations.items()}
        next_embeddings = self.next_embeddings[batch_inds, env_indices]

        # Sample and set new goals
        new_goals, goal_embeddings = self._sample_goals(batch_inds, env_indices)
        obs["goal"] = new_goals
        # The goal for the next observation must be the same as the previous one. TODO: Why ?
        next_obs["goal"] = new_goals

        # Compute new reward
        dist = np.linalg.norm(goal_embeddings - next_embeddings, axis=1)
        is_success = dist < self.distance_threshold
        rewards = is_success.astype(np.float32) - 1
        for idx, info in enumerate(self.infos[batch_inds, env_indices]):
            if info.get("dead", False):
                rewards[idx] -= 100

        obs = self._normalize_obs(obs)
        next_obs = self._normalize_obs(next_obs)

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])),
            rewards=self.to_torch(self._normalize_reward(rewards, env)),
        )

    def _sample_goals(self, batch_inds: np.ndarray, env_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample goals based on goal_selection_strategy.

        :param batch_inds: Indices of the transitions
        :param env_indices: Indices of the envrionments
        :return: Goals and their corresponding latent representation
        """
        batch_ep_start = self.ep_start[batch_inds, env_indices]
        batch_ep_length = self.ep_length[batch_inds, env_indices]

        if self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # replay with final state of current episode
            transition_indices_in_episode = batch_ep_length - 1

        elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # replay with random state which comes from the same episode and was observed after current transition
            current_indices_in_episode = batch_inds - batch_ep_start
            transition_indices_in_episode = np.random.randint(current_indices_in_episode, batch_ep_length)

        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # replay with random state which comes from the same episode as current transition
            transition_indices_in_episode = np.random.randint(0, batch_ep_length)

        else:
            raise ValueError(f"Strategy {self.goal_selection_strategy} for sampling goals not supported!")

        transition_indices = (transition_indices_in_episode + batch_ep_start) % self.buffer_size
        return (
            self.next_observations["observation"][transition_indices, env_indices],
            self.next_embeddings[transition_indices, env_indices],
        )
