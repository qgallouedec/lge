from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from gym import spaces
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

from lge.inverse_model import InverseModel
from lge.utils import estimate_density, is_image, sample_geometric_with_max
from stable_baselines3.common.vec_env import VecTransposeImage


class ArchiveBuffer(HerReplayBuffer):
    """
    Archive buffer.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param env: The training environment
    :param inverse_model: Inverse model used to compute embeddings
    :param distance_threshold: The goal is reached when the distance between the current embedding
        and the goal embedding is under this threshold
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param n_sampled_goal: Number of virtual transitions to create per real transition,
        by sampling new goals.
    :param goal_selection_strategy: Strategy for sampling goals for replay.
        One of ['episode', 'final', 'future']
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        env: VecEnv,
        inverse_model: InverseModel,
        distance_threshold: float = 1.0,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        n_sampled_goal: int = np.inf,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
    ) -> None:
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            env,
            device=device,
            n_envs=n_envs,
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy=goal_selection_strategy,
        )

        self.distance_threshold = distance_threshold
        self.inverse_model = inverse_model
        emb_dim = inverse_model.latent_size

        self.goal_embeddings = np.zeros((self.buffer_size, self.n_envs, emb_dim), dtype=np.float32)
        self.next_embeddings = np.zeros((self.buffer_size, self.n_envs, emb_dim), dtype=np.float32)

        # The archive does not compute embedding of every new transition stored. The embeddings are
        # computed when the method recompute_embeddings() is appealed. To keep track of the number
        # embedding computed, we use self.self.nb_embeddings_computed
        self.nb_embeddings_computed = 0

    def recompute_embeddings(self) -> None:
        """
        Re-compute all the embeddings and estiamte the density. This method must
        be called on a regular basis to keep the density estimation up to date.
        """
        upper_bound = self.pos if not self.full else self.buffer_size

        for env_idx in range(self.n_envs):
            # Recompute 256 by 256 to avoid cuda space allocation error.
            k = 0
            while k < upper_bound:
                upper = min(upper_bound, k + 256)
                goal_embedding = self.encode(self.observations["goal"][k:upper][env_idx])
                self.goal_embeddings[k:upper][env_idx] = goal_embedding.detach().cpu().numpy()

                next_embedding = self.encode(self.next_observations["observation"][k:upper][env_idx])
                self.next_embeddings[k:upper][env_idx] = next_embedding.detach().cpu().numpy()
                k += 256

        self.nb_embeddings_computed = upper_bound * self.n_envs

        # Reshape and convert embeddings to torch tensor
        all_embeddings = self.next_embeddings[:upper_bound]
        all_embeddings = all_embeddings.reshape(self.nb_embeddings_computed, -1)
        all_embeddings = self.to_torch(all_embeddings)

        # Estimate density based on the embeddings
        density = np.zeros((self.nb_embeddings_computed), dtype=np.float32)
        k = 0
        while k < self.nb_embeddings_computed:
            upper = min(self.nb_embeddings_computed, k + 256)
            embeddings = all_embeddings[k:upper]
            density[k:upper] = estimate_density(embeddings, all_embeddings).detach().cpu().numpy()
            k += 256

        self.sorted_density = np.argsort(density)

    def encode(self, obs: np.ndarray) -> torch.Tensor:
        obs = self.to_torch(obs).float()
        if is_image(obs):
            # Convert all to float
            assert torch.max(obs) > 1
            obs = obs / 255
            if obs.shape[-1] == 3:
                obs = VecTransposeImage.transpose_image(obs)
            if len(obs.shape) == 3:
                obs = obs.unsqueeze(0)
        self.inverse_model.eval()
        return self.inverse_model.encoder(obs)

    def sample_trajectory(self, step: int = 1) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Sample a trajcetory of observations based on the embeddings density.

        :return: A list of observations as array
        """

        if self.nb_embeddings_computed == 0:  # no embeddings computed yet
            goal = np.expand_dims(self.observation_space["goal"].sample(), 0)
            return goal, self.encode(goal).detach().cpu().numpy()

        goal_id = self.sorted_density[sample_geometric_with_max(0.01, max_value=self.sorted_density.shape[0]) - 1]
        goal_pos = goal_id // self.n_envs
        goal_env = goal_id % self.n_envs
        start = self.ep_start[goal_pos, goal_env]
        trajectory = self.next_observations["observation"][start : goal_pos + 1, goal_env]
        emb_trajectory = self.next_embeddings[start : goal_pos + 1, goal_env]
        trajectory, emb_trajectory = np.flip(trajectory[::-step], 0), np.flip(emb_trajectory[::-step], 0)
        return trajectory, emb_trajectory

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
        obs = {key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}
        next_obs = {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}
        next_embeddings = self.next_embeddings[batch_inds, env_indices, :]

        # Sample and set new goals
        new_goals, goal_embeddings = self._sample_goals(batch_inds, env_indices)
        obs["goal"] = new_goals
        # The goal for the next observation must be the same as the previous one. TODO: Why ?
        next_obs["goal"] = new_goals

        # Compute new reward
        dist = np.linalg.norm(goal_embeddings - next_embeddings, axis=1)
        is_success = dist < self.distance_threshold
        rewards = is_success.astype(np.float32) - 1

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
            dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(rewards.reshape(-1, 1), env)),
        )

    def _sample_goals(self, batch_inds: np.ndarray, env_indices: np.ndarray) -> np.ndarray:
        """
        Sample goals based on goal_selection_strategy.

        :param trans_coord: Coordinates of the transistions within the buffer
        :return: Return sampled goals
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
