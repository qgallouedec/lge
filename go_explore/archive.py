import copy
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.goal_selection_strategy import KEY_TO_GOAL_STRATEGY, GoalSelectionStrategy

from go_explore.cells import CellFactory
from go_explore.utils import index, multinomial


class ArchiveBuffer(DictReplayBuffer):
    """
    Archive buffer.

    - HER sampling
    - Sample trajectory of observations based on cells
    When you change the cell_factory, you need to call `when_cell_factory_updated`.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param cell_factory: The cell factory
    :param count_pow: The goal cell is sampled with weight is 1 / count**count_pow
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
        cell_factory: CellFactory,
        count_pow: float = 0,
        device: Union[th.device, str] = "cpu",
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
        self.count_pow = count_pow
        self.infos = np.array([[{} for _ in range(self.n_envs)] for _ in range(self.buffer_size)])

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

        # For cell management
        self.cell_factory = cell_factory
        self._reset_cell_trackers()

    def _reset_cell_trackers(self) -> None:
        cell_shape = self.cell_factory.cell_space.shape
        # self.counts maps cell uid to the cell visitation count.
        self.counts = np.empty((0,), dtype=np.int64)
        # self.earliest_cell_envs maps cell uid to the env index of the earliest cell visitation.
        self.earliest_cell_env = np.empty((0,), dtype=np.int64)
        # self.earliest_cell_pos maps cell uid to the buffer position of the earliest cell visitation.
        self.earliest_cell_pos = np.empty((0,), dtype=np.int64)
        # self.cells maps buffer position to the cell representation.
        self.cells = np.zeros((self.buffer_size, self.n_envs, *cell_shape))
        # self.unique_cells maps cell uid to its cell representation.
        self.unique_cells = np.empty((0, *cell_shape), dtype=self.cell_factory.cell_space.dtype)

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
        self._add_new_cell(next_obs["observation"])

        # Update episode start
        self.ep_start[self.pos] = self._current_ep_start.copy()

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

    def _add_new_cell(self, obs: np.ndarray) -> None:
        """
        Process an observation and update the cell count and the trajectories.

        :param obs: The observation.
        :type obs: np.ndarray
        """
        cells = self.compute_cell(obs)
        self.cells[self.pos] = cells
        for env_idx in range(self.n_envs):
            cell = cells[env_idx]
            maybe_cell_uid = index(cell, self.unique_cells)
            if maybe_cell_uid is not None:
                # Cell is known, so we increase the cell count
                self.counts[maybe_cell_uid] += 1
                # Get the min known distance
                best_pos, best_env = self.earliest_cell_pos[maybe_cell_uid], self.earliest_cell_env[maybe_cell_uid]
                min_distance_to_cell = best_pos - self.ep_start[best_pos, best_env]
                current_distance_to_cell = self.pos - self._current_ep_start[env_idx]
                # If this time we've reached the cell earlier, update the attributes
                if current_distance_to_cell < min_distance_to_cell:
                    self.earliest_cell_env[maybe_cell_uid] = env_idx
                    self.earliest_cell_pos[maybe_cell_uid] = self.pos
            else:
                # The cell is new
                self.unique_cells = np.concatenate((self.unique_cells, np.expand_dims(cell, axis=0)))
                self.counts = np.concatenate((self.counts, [1]))
                self.earliest_cell_env = np.concatenate((self.earliest_cell_env, [env_idx]))
                self.earliest_cell_pos = np.concatenate((self.earliest_cell_pos, [self.pos]))

    def compute_cell(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute the cell of the observation.

        :param obs: The observation as an array.
        :return: The cell, as an array
        """
        th_obs = self.to_torch(obs)
        cells = self.cell_factory(th_obs).cpu().numpy()
        return cells

    def when_cell_factory_updated(self) -> None:
        """
        Call this function when you change the parametrisation of the cell factory.
        It computes the new cells and the new traejctories.
        """
        self._reset_cell_trackers()
        self._recompute_cells()
        self._recompute_trajectories()

    def _recompute_cells(self) -> None:
        """
        Re-compute all the cells.
        """
        upper_bound = self.pos if not self.full else self.buffer_size
        # Recompute 256 by 256 to avoid cuda space allocation error.
        k = 0
        while k < upper_bound:
            upper = min(upper_bound, k + 256)
            self.cells[k:upper] = self.compute_cell(self.next_observations["observation"][k:upper])
            k += 256

    def _recompute_trajectories(self) -> None:
        """
        Update the trajectories based on the cells. Must be called after updating cells.
        """
        upper_bound = self.pos if not self.full else self.buffer_size
        cells = self.to_torch(self.cells[:upper_bound])
        nb_obs = upper_bound * self.n_envs
        if upper_bound == 0:
            return  # no trajectory yet
        flat_cells = cells.reshape((nb_obs, -1))  # shape from (pos, env_idx, *cell_shape) to (idx, *cell_shape)
        # Compute the unique cells.
        # cells_uid is a tensor of shape (nb_obs,) mapping observation index to its cell index.
        # unique_cells is a tensor of shape (nb_cells, *cell_shape) mapping cell index to the cell.
        unique_cells, cells_uid, counts = th.unique(flat_cells, return_inverse=True, return_counts=True, dim=0)
        self.counts = counts.cpu().numpy()  # type: np.ndarray
        self.unique_cells = unique_cells.cpu().numpy()  # type: np.ndarray
        nb_cells = self.unique_cells.shape[0]  # number of unique cells
        flat_pos = th.arange(self.ep_start.shape[0]).repeat_interleave(self.n_envs)  # [0, 0, 1, 1, 2, ...] if n_envs == 2
        flat_ep_start = th.from_numpy(self.ep_start).flatten()  # shape from (pos, env_idx) to (idx,)
        flat_timestep = flat_pos - flat_ep_start  # timestep within the episode

        earliest_cell_occurence = th.zeros(nb_cells, dtype=th.int64)
        for cell_uid in range(nb_cells):
            cell_idxs = th.where(cell_uid == cells_uid)[0]  # index of observations that are in the cell
            all_cell_occurences_timestep = flat_timestep[cell_idxs]  # the cell has been visited after all these timesteps
            earliest = th.argmin(all_cell_occurences_timestep)  # focus on the time when the cell is visited the earliest
            earliest_cell_occurence[cell_uid] = cell_idxs[earliest]
        self.earliest_cell_env = (earliest_cell_occurence % self.n_envs).cpu().numpy()
        self.earliest_cell_pos = th.div(earliest_cell_occurence, self.n_envs, rounding_mode="floor").cpu().numpy()

    def sample_trajectory(self) -> List[np.ndarray]:
        """
        Sample a trajcetory of observations based on the cells counts and trajectories.

        A goal cell is sampled with weight 1/count**count_pow. Then the shortest
        trajectory to the cell is computed and returned.

        :return: A list of observations as array
        """
        if self.counts.shape[0] == 0:  # no cells yet
            goal = self.observation_space["goal"].sample()
            return [goal]
        # Weights depending of the cell visitation count
        weights = 1 / np.sqrt(self.counts + 1)
        cell_uid = multinomial(weights)
        # Get the env_idx, the pos in the buffer and the position of the start of the trajectory
        env = self.earliest_cell_env[cell_uid]
        goal_pos = self.earliest_cell_pos[cell_uid]
        start = self.ep_start[goal_pos, env]
        # Loop to avoid consecutive repetition
        trajectory = [self.next_observations["observation"][start, env]]
        for pos in range(start + 1, goal_pos + 1):
            previous_cell = self.cells[pos - 1, env]
            cell = self.cells[pos, env]
            if (previous_cell != cell).any():
                obs = self.next_observations["observation"][pos, env]
                trajectory.append(obs)
        return trajectory

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
            key: th.cat((real_data.observations[key], virtual_data.observations[key]))
            for key in virtual_data.observations.keys()
        }
        actions = th.cat((real_data.actions, virtual_data.actions))
        next_observations = {
            key: th.cat((real_data.next_observations[key], virtual_data.next_observations[key]))
            for key in virtual_data.next_observations.keys()
        }
        dones = th.cat((real_data.dones, virtual_data.dones))
        rewards = th.cat((real_data.rewards, virtual_data.rewards))

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

        if her_relabeling:
            infos = copy.deepcopy(self.infos[batch_inds, env_indices])
            # Sample and set new goals
            new_goals = self._sample_goals(batch_inds, env_indices)
            obs["goal"] = new_goals
            # The goal for the next observation must be the same as the previous one. TODO: Why ?
            next_obs["goal"] = new_goals
            # The goal has changed, there is no longer a guarantee that the transition is
            # successful. Since it is not possible to easily get this information, we prefer
            # to remove it. The success information is not used in the learning algorithm anyway.
            for info in infos:
                info.pop("is_success", None)
            # Compute new reward
            rewards = self.env.env_method(
                "compute_reward",
                # here we use the new goal
                obs["goal"],
                # the new state depends on the previous state and action
                # s_{t+1} = f(s_t, a_t)
                # so the next observation depends also on the previous state and action
                # because we are in a GoalEnv:
                # r_t = reward(s_t, a_t) = reward(next_obs, goal)
                # therefore we have to use next_obs["observation"] and not obs["observation"]
                next_obs["observation"],
                infos,
                # we use the method of the first environment assuming that all environments are identical.
                indices=[0],
            )
            rewards = rewards[0].astype(np.float32)  # env_method returns a list containing one element
        else:
            rewards = self.rewards[batch_inds, env_indices]

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
        return self.next_observations["observation"][transition_indices, env_indices]

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
