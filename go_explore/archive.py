import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from gym import spaces
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.goal_selection_strategy import KEY_TO_GOAL_STRATEGY, GoalSelectionStrategy
from go_explore.inverse_model import InverseModel

from go_explore.utils import multinomial, estimate_density


class ArchiveBuffer(DictReplayBuffer):
    """
    Archive buffer.

    - HER sampling
    - Sample trajectory of observations embedding density

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param distance_threshold: when the current state and the goa state are under this distance in
        latent space, the agent gets a reward
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
        :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
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
        inverse_model: InverseModel,
        distance_threshold: float = 1.0,
        device: Union[torch.device, str] = "cpu",
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
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.env = None
        self.n_sampled_goal = n_sampled_goal
        # compute ratio between HER replays and regular replays in percent for online HER sampling
        self.her_ratio = 1 - (1.0 / (self.n_sampled_goal + 1))
        self.infos = np.array([[{} for _ in range(self.n_envs)] for _ in range(self.buffer_size)])

        self.distance_threshold = distance_threshold

        if isinstance(goal_selection_strategy, str):
            goal_selection_strategy = KEY_TO_GOAL_STRATEGY[goal_selection_strategy.lower()]
        # check if goal_selection_strategy is valid
        assert isinstance(
            goal_selection_strategy, GoalSelectionStrategy
        ), f"Invalid goal selection strategy, please use one of {list(GoalSelectionStrategy)}"
        self.goal_selection_strategy = goal_selection_strategy

        # For trajectory handling
        self.ep_start = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self._current_ep_start = np.zeros(self.n_envs, dtype=np.int64)
        self.ep_length = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)

        self.inverse_model = inverse_model
        emb_dim = inverse_model.latent_size
        self.embeddings = np.zeros((self.buffer_size, self.n_envs, emb_dim), dtype=np.float32)
        self.goal_embeddings = np.zeros((self.buffer_size, self.n_envs, emb_dim), dtype=np.float32)
        self.next_embeddings = np.zeros((self.buffer_size, self.n_envs, emb_dim), dtype=np.float32)
        self.next_goal_embeddings = np.zeros((self.buffer_size, self.n_envs, emb_dim), dtype=np.float32)
        self.density = np.zeros((self.buffer_size * self.n_envs), dtype=np.float32)
        self.embedding_computed = 0

    def __getstate__(self) -> Dict[str, Any]:
        """
        Gets state for pickling.
        Excludes self.env, as in general Env's may not be pickleable.
        Note: when using offline sampling, this will also save the offline replay buffer.
        """
        state = self.__dict__.copy()
        # these attributes are not pickleable
        del state["env"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restores pickled state.
        User must call ``set_env()`` after unpickling before using.
        :param state:
        """
        self.__dict__.update(state)
        assert "env" not in state
        self.env = None

    def set_env(self, env: VecEnv) -> None:
        """
        Sets the environment.
        :param env:
        """
        if self.env is not None:
            raise ValueError("Trying to set env of already initialized environment.")

        self.env = env

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        self._erase_if_write_over()
        # Update episode start
        self.ep_start[self.pos] = self._current_ep_start

        # Store the transition
        self.infos[self.pos] = infos
        super().add(obs, next_obs, action, reward, done, infos)

        # When episode ends, compute and store the episode length.
        # It need to be done after adding in order to self.pos to be updated.
        for env_idx in range(self.n_envs):
            if done[env_idx]:
                episode_start = self._current_ep_start[env_idx]
                episode_end = self.pos
                if episode_end < episode_start:
                    # Occurs when the buffer becomes full, the storage resumes at the
                    # beginning of the buffer. This can happen in the middle of an episode.
                    episode_end += self.buffer_size
                episode = np.arange(episode_start, episode_end) % self.buffer_size
                self.ep_length[episode, env_idx] = episode_end - episode_start
                # Update the current episode start
                self._current_ep_start[env_idx] = self.pos

    def _erase_if_write_over(self) -> None:
        """
        Set the lenght of the episode we are about to write on to zero.
        """
        # When the buffer is full, we write over old episodes. When we start to
        # rewrite on an old episodes, we want the whole old episode to be deleted
        # (and not only the transition on which we rewrite). To do this, we set
        # the length of the old episode to 0, so it can't be sampled anymore.
        for env_idx in range(self.n_envs):
            episode_length = self.ep_length[self.pos][env_idx]
            if episode_length > 0:
                episode_end = self.ep_start[self.pos][env_idx] + episode_length
                episode_indices = np.arange(self.pos, episode_end) % self.buffer_size
                self.ep_length[episode_indices, env_idx] = 0

    def recompute_embeddings(self) -> None:
        """
        Re-compute all the embeddings.
        """
        upper_bound = self.pos if not self.full else self.buffer_size
        # Recompute 256 by 256 to avoid cuda space allocation error.
        k = 0
        while k < upper_bound:
            upper = min(upper_bound, k + 256)
            self.embeddings[k:upper] = self.encode(self.observations["observation"][k:upper]).detach().cpu().numpy()
            self.goal_embeddings[k:upper] = self.encode(self.observations["goal"][k:upper]).detach().cpu().numpy()
            self.next_embeddings[k:upper] = self.encode(self.next_observations["observation"][k:upper]).detach().cpu().numpy()
            self.next_goal_embeddings[k:upper] = self.encode(self.next_observations["goal"][k:upper]).detach().cpu().numpy()
            k += 256

        all_embeddings = self.next_embeddings[:upper_bound]
        all_embeddings = all_embeddings.reshape(upper_bound * self.n_envs, -1)
        all_embeddings = self.to_torch(all_embeddings)
        k = 0
        while k < upper_bound:
            upper = min(upper_bound, k + 256)
            embeddings = all_embeddings[k:upper]
            density = estimate_density(embeddings, all_embeddings).detach().cpu().numpy()
            self.density[k:upper] = density
            k += 256

        self.embedding_computed = upper_bound

    def encode(self, obs: np.ndarray) -> torch.Tensor:
        obs = self.to_torch(obs).float()
        self.inverse_model.eval()
        return self.inverse_model.encoder(obs)

    def sample_trajectory(self, density_pow: float = 0.0, step: int = 1) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Sample a trajcetory of observations based on the embeddings density.

        A goal is sampled with weight density**density_pow.

        :return: A list of observations as array
        """

        if self.embedding_computed == 0:  # no embeddings computed yet
            goal = np.expand_dims(self.observation_space["goal"].sample(), 0)
            return goal, self.encode(goal).detach().cpu().numpy()

        density = self.to_torch(self.density[: self.embedding_computed])
        weights = torch.pow(density, density_pow)
        goal_id = multinomial(weights)
        goal_pos = torch.div(goal_id, self.n_envs, rounding_mode="floor").cpu().numpy()
        goal_env = (goal_id % self.n_envs).cpu().numpy()
        start = self.ep_start[goal_pos, goal_env]
        trajectory = self.next_observations["observation"][start : goal_pos + 1, goal_env]
        emb_trajectory = self.next_embeddings[start : goal_pos + 1, goal_env]
        trajectory, emb_trajectory = np.flip(trajectory[::-step], 0), np.flip(emb_trajectory[::-step], 0)
        return trajectory, emb_trajectory

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer. Part of the returned observations are relabeled with HER.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv to normalize the observations/rewards when sampling
        :return: Samples
        """
        env_indices = np.random.randint(self.n_envs, size=batch_size)
        batch_inds = np.zeros_like(env_indices)

        # When the buffer is full, we rewrite on old episodes. We don't want to
        # sample incomplete episode transitions, so we have to eliminate some indexes.
        is_valid = self.ep_length > 0

        # Sample batch indices from the valid indices
        valid_inds = [np.arange(self.buffer_size)[is_valid[:, env_idx]] for env_idx in range(self.n_envs)]
        for i, env_idx in enumerate(env_indices):
            batch_inds[i] = np.random.choice(valid_inds[env_idx])

        # Split the indexes between real and virtual transitions.
        nb_virtual = int(self.her_ratio * batch_size)
        virtual_batch_inds, real_batch_inds = np.split(batch_inds, [nb_virtual])
        virtual_env_indices, real_env_indices = np.split(env_indices, [nb_virtual])

        # get real and virtual data
        real_data = self._get_samples(real_batch_inds, real_env_indices, her_relabeling=False, env=env)
        virtual_data = self._get_samples(virtual_batch_inds, virtual_env_indices, her_relabeling=True, env=env)

        # Concatenate real and virtual data
        observations = {
            key: torch.cat((real_data.observations[key], virtual_data.observations[key]))
            for key in virtual_data.observations.keys()
        }
        actions = torch.cat((real_data.actions, virtual_data.actions))
        next_observations = {
            key: torch.cat((real_data.next_observations[key], virtual_data.next_observations[key]))
            for key in virtual_data.next_observations.keys()
        }
        dones = torch.cat((real_data.dones, virtual_data.dones))
        rewards = torch.cat((real_data.rewards, virtual_data.rewards))

        return DictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
        )

    def _get_samples(
        self, batch_inds: np.ndarray, env_indices: np.ndarray, her_relabeling: bool, env: Optional[VecNormalize] = None
    ) -> DictReplayBufferSamples:
        """
        Get the samples corresponding to the batch and environment indices.

        :param batch_inds: Indices of the transitions
        :param env_indices: Indices of the envrionments
        :param her_relabeling: If True, sample new goals and compute new rewards with HER.
        :param env: associated gym VecEnv to normalize the observations/rewards when sampling, defaults to None
        :return: Samples
        """
        # Get infos and obs
        obs = {key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}
        next_obs = {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}
        next_embeddings = self.next_embeddings[batch_inds, env_indices, :]
        goal_embeddings = self.goal_embeddings[batch_inds, env_indices, :]
        if her_relabeling:
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

    def truncate_last_trajectory(self) -> None:
        """
        Only for online sampling, called when loading the replay buffer.
        If called, we assume that the last trajectory in the replay buffer was finished
        (and truncate it).
        If not called, we assume that we continue the same trajectory (same episode).
        """
        # If we are at the start of an episode, no need to truncate
        if (self.ep_start[self.pos] != self.pos).any():
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated.\n"
                "If you are in the same episode as when the replay buffer was saved,\n"
                "you should use `truncate_last_trajectory=False` to avoid that issue."
            )
            self.ep_start[-1] = self.pos
            # set done = True for current episodes
            self.dones[self.pos - 1] = True
