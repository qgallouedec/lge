import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
import torch as th
from gym import spaces
from scipy.sparse.csgraph import shortest_path
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples, ReplayBufferSamples
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from go_explore.go_explore.cell_computers import Cell, CellComputer


class RewardPrioritizedReplayBuffer(ReplayBuffer):
    """
    Replay buffer used for task learning. Samples such that mean rewards is aroud 0.5.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
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
    """

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        max_idx = self.buffer_size if self.full else self.pos
        all_idxs = np.arange(max_idx)
        r = self.rewards[:, 0][:max_idx]
        weights = (r == 1) / (r == 1).sum() + (r == 0) / (r == 0).sum()
        # weights = self.rewards[:, 0][:max_idx] + self.rewards.mean()
        p = weights / weights.sum()
        # randomly choose a final cell
        batch_inds = np.random.choice(all_idxs, p=p, size=batch_size)
        return self._get_samples(batch_inds, env=env)


class ArchiveBuffer(ReplayBuffer):
    """
    ReplayBuffer that keep track of cells.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param cell_computer: The cell computer
    :param count_pow:
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
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        cell_computer: CellComputer,
        count_pow: int = 0,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
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
        self.cell_computer = cell_computer
        self.count_pow = count_pow

        self._cell_to_idx = {}  # A dict mapping cell to a unique idx
        self._idx_to_cell = []  # Same, but the other way. Faster than using .index()
        # A dict mapping cell to a list of every encountered observation in that cell
        self._cell_to_obss: Dict[Cell, List[np.ndarray]] = {}
        self.nb_cells = 0  # The number of encountered cells
        # csgraph is a matrix to store idx that are neighboors (csgraph[5][2] == 1 means that 5 can lead to 2)
        self.csgraph = np.zeros(shape=(0, 0))
        self._counts = np.zeros(shape=(0,))  # A counter of the number of visits per cell

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Add an element to the buffer.

        :param obs: the current observation
        :param next_obs: the next observation
        :param action: the action
        :param reward: the reward
        :param done: whether the env is done
        :param infos: infos
        :param episode_start: whether the episode starts, defaults to False
        """
        super().add(obs, next_obs, action, reward, done, infos)
        for _obs, _next_obs, _infos in zip(obs, next_obs, infos):
            self._process_transition(_obs, _next_obs, _infos)

    def _process_transition(self, obs: np.ndarray, next_obs: np.ndarray, infos: Dict[str, Any]) -> None:
        # compute cells
        current_cell = self.cell_computer.compute_cell(obs)
        next_cell = self.cell_computer.compute_cell(next_obs)
        # we consider the current cell only if the episode starts
        # if not, it means that the current has already been processed
        if infos.get("episode_start", False):
            self._update_counts(obs, current_cell)
        self._update_counts(next_obs, next_cell)
        self._update_csgraph(current_cell, next_cell)

    def _update_counts(self, obs: np.ndarray, cell: Cell) -> None:
        if cell not in self._cell_to_idx:
            self._new_cell_found(cell)
        idx = self._cell_to_idx[cell]
        # update counts and obs list
        self._counts[idx] += 1
        self._cell_to_obss[cell].append(obs)

    def _new_cell_found(self, cell: Cell) -> None:
        """
        Call this when you have found a new cell.

        It expands the csgraph, counts, cell_to_obss, cell_to_idx and idx_to_cell
        """
        idx = len(self._cell_to_idx)
        self._idx_to_cell.append(cell)
        self._cell_to_idx[cell] = idx
        self._cell_to_obss[cell] = []
        self.nb_cells += 1
        # expanding arrays
        self._counts = np.pad(self._counts, (0, 1), constant_values=0)
        self.csgraph = np.pad(self.csgraph, ((0, 1), (0, 1)), constant_values=np.inf)

    def _update_csgraph(self, current_cell: Cell, next_cell: Cell) -> None:

        # update csgraph
        current_cell_idx = self._cell_to_idx[current_cell]
        next_cell_idx = self._cell_to_idx[next_cell]
        self.csgraph[current_cell_idx][next_cell_idx] = min(self.csgraph[current_cell_idx][next_cell_idx], 1)

    def sample_subgoal_path(self, from_obs: np.ndarray, subgoal_horizon: int = 1) -> List[np.ndarray]:
        """
        Samples a subgoal path that starts from the given observation.

        First, compute all reachable observations from the given observation (based on the trajectories already encountered).
        Second, samples a final observation from these reachable observations. The less a final observation has been visited,
        the more likely it is to be sampled.
        Third, uses a shortest path algorithm to select intermediate observations for reaching this final observation.

        :param from_obs: The observation taken as a starting point
        :param subgoal_horizon: Number of cells separating two observations in the subgoal path
        :return: The subgoal_path of observations.
        """
        # compute the initial cell
        from_cell = self.cell_computer.compute_cell(from_obs)
        from_idx = self._cell_to_idx[from_cell]
        # process the dist matrix, to determine the reachable cells.
        dist_matrix, predecessors = shortest_path(self.csgraph, return_predecessors=True)
        reachable_idxs = self._get_reachable_idxs(from_idx, dist_matrix)
        if len(reachable_idxs) == 0:  # if there is no reacheable cells
            return [np.copy(from_obs)]
        # compute the weights associated with these reachable cells.
        weights = self._get_weights(reachable_idxs)
        p = weights / weights.sum()
        # randomly choose a final cell
        to_idx = np.random.choice(reachable_idxs, p=p)
        subgoal_idx_path = self._get_path(predecessors, from_idx, to_idx)
        subgoal_idx_path.pop(0)  # no need to take the current cell
        # convert cells into observations
        cell_path = [self._idx_to_cell[idx] for idx in subgoal_idx_path]
        subgoal_path = [self._cell_to_obs(cell) for cell in cell_path]
        subgoal_path = self._lighten_path(subgoal_path, subgoal_horizon)
        return subgoal_path

    def solve_task(
        self, from_obs: np.ndarray, task: Callable[[np.ndarray], np.ndarray], subgoal_horizon: int
    ) -> List[np.ndarray]:
        """
        Plan the shortest path to realise the given task

        :param from_obs: The starting observation
        :param task: The function used to determine whether an observation is a success or not
        :param subgoal_horizon: Number of cells separating two observations in the subgoal path
        :return: A path of subgoals to solve the task
        """
        # compute the initial cell
        from_cell = self.cell_computer.compute_cell(from_obs)
        from_idx = self._cell_to_idx[from_cell]
        # process the dist matrix, to determine the reachable cells.
        dist_matrix, predecessors = shortest_path(self.csgraph, return_predecessors=True)
        reachable_idxs = self._get_reachable_idxs(from_idx, dist_matrix)
        distances = dist_matrix[from_idx][reachable_idxs]
        # compute the reacheable observation: strong assumption here : we consider all obs of
        # a cell have the same state wrt to teh success
        reachable_cells = [self._idx_to_cell[idx] for idx in reachable_idxs]
        reachable_obs = np.array([self._cell_to_obs(cell) for cell in reachable_cells], dtype=np.float32)
        # compute the reward (is_success actually)
        is_success = task(reachable_obs).squeeze()
        # only consider the observations, distance that correspond to a success
        distances = distances[is_success]
        reachable_idxs = reachable_idxs[is_success]
        # take the min distance, and get the target_obs
        idx = np.argmin(distances)
        to_idx = reachable_idxs[idx]
        # compute the path toward the target_obs and return
        subgoal_idx_path = self._get_path(predecessors, from_idx, to_idx)
        # convert cells into observations
        cell_path = [self._idx_to_cell[idx] for idx in subgoal_idx_path]
        subgoal_path = [self._cell_to_obs(cell) for cell in cell_path]
        subgoal_path = self._lighten_path(subgoal_path, subgoal_horizon)
        return subgoal_path

    def _get_reachable_idxs(self, from_idx: int, dist_matrix: np.ndarray) -> List[int]:
        """
        Get the list of the reachable indexes from the given index and the distance matrix.

        :param from_idx: The intial index
        :param dist_matrix: The distance matrix
        :return: A list of idexes, corresponding to the list of indexes reachable.
        """
        # just focus on the given cell
        dist_to_other_cells = dist_matrix[from_idx]
        # take the indices of the cells for which the distance is not infinite.
        arange = np.arange(self.nb_cells)
        # compute the reacheable idexes.
        # first condition : distance < inf
        # * for "and"
        # second condition : I don't care about staying in the same cell
        reachable_idxs = arange[(dist_to_other_cells < np.inf) * (dist_to_other_cells > 0)]
        # convert into cells
        return reachable_idxs

    def _get_weights(self, reachable_idxs: List[int]) -> np.ndarray:
        """
        Return the list of the weigths associated with each index in the list.

        :param reachable_idxs: The list of indexes for which we want the weights
        :return: The list of weights associated with the input indexes
        """
        # The more count, the less weight. See go-explore paper formula.
        weights = 1 / (self._counts ** self.count_pow)
        # take only the weigts of the reacheable cells
        reachable_weights = weights[reachable_idxs]
        return reachable_weights

    def _get_path(self, predecessors: np.ndarray, from_idx: int, to_idx: int) -> List[int]:
        """
        Get the shortest path from an integer to another, based on the predecessors matrix.

        :param predecessors: The predecessor matrix
        :param from_idx: The initial integer
        :param to_idx: The target integer
        :return: The shortest path
        """
        path = [to_idx]
        current_idx = to_idx
        while predecessors[from_idx, current_idx] != -9999:
            path.append(predecessors[from_idx, current_idx])
            current_idx = predecessors[from_idx, current_idx]
        path.reverse()
        return path

    def _cell_to_obs(self, cell: Cell) -> np.ndarray:
        """
        Randomly choose an observation among the observations of the given cell.

        :param cell: The cell
        """
        obs = random.choice(self._cell_to_obss[cell])
        return obs

    def _lighten_path(self, path: List[np.ndarray], subgoal_horizon: int) -> List[np.ndarray]:
        """
        Pick a list of elements from the list, evenly spaced by a certain number of steps, keeping the last one.

        Example:
        >>> _lighten_path([1, 2, 3, 4, 5, 6], subgoal_horizon=3)
        [3, 6]

        :param path: The path
        :param subgoal_horizon: Number of cells separating two observations in the subgoal path
        :return: The lightened path
        """
        path = path[::-subgoal_horizon]
        path.reverse()
        return path

    def copy(self) -> ReplayBuffer:
        buffer = RewardPrioritizedReplayBuffer(self.buffer_size, self.observation_space, self.action_space)
        buffer.actions = self.actions.copy()
        buffer.observations = self.observations
        buffer.next_observations = self.next_observations
        buffer.actions = self.actions
        buffer.rewards = self.rewards
        buffer.dones = self.dones
        buffer.timeouts = self.timeouts
        buffer.pos = self.pos
        buffer.full = self.full
        return buffer


class PrioritizedHerDictReplayBuffer(HerReplayBuffer, ArchiveBuffer):
    def __init__(
        self,
        env: VecEnv,
        buffer_size: int,
        cell_computer: CellComputer,
        count_pow: int = 0,
        device: Union[th.device, str] = "cpu",
        replay_buffer: Optional[DictReplayBuffer] = None,
        max_episode_length: Optional[int] = None,
        n_sampled_goal: int = 4,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
        online_sampling: bool = True,
        handle_timeout_termination: bool = True,
    ):
        HerReplayBuffer.__init__(
            self,
            env,
            buffer_size,
            device=device,
            replay_buffer=replay_buffer,
            max_episode_length=max_episode_length,
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy=goal_selection_strategy,
            online_sampling=online_sampling,
            handle_timeout_termination=handle_timeout_termination,
        )
        ArchiveBuffer.__init__(
            self,
            buffer_size,
            env.observation_space["observation"],
            env.action_space,
            cell_computer,
            count_pow=count_pow,
            device=device,
            n_envs=1,
            optimize_memory_usage=False,
            handle_timeout_termination=handle_timeout_termination,
        )
        self._cell_to_transitions = {}

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        for _obs, _next_obs, _infos in zip(obs["observation"], next_obs["observation"], infos):
            self._process_transition(_obs, _next_obs, _infos)
        super().add(obs, next_obs, action, reward, done, infos)

    def _cell_to_transition(self, cell):
        episode_idx, transition_idx = random.choice(self._cell_to_transitions[cell])
        ep_length = self.episode_lengths[episode_idx]
        if transition_idx == ep_length-1:
            transition_idx -= 1
        return episode_idx, transition_idx

    def _new_cell_found(self, cell: Cell) -> None:
        super()._new_cell_found(cell)
        self._cell_to_transitions[cell] = []

    def _update_counts(self, obs: np.ndarray, cell: Cell) -> None:
        super()._update_counts(obs, cell)
        # a little bit incorrect for the first transition. It does not cont
        # transition but cell visition. It might not be a problem when agent
        # always start from the same cell.
        self._cell_to_transitions[cell].append((self.pos, self.current_idx))

    def _sample_transitions(
        self,
        batch_size: Optional[int],
        maybe_vec_env: Optional[VecNormalize],
        online_sampling: bool,
        n_sampled_goal: Optional[int] = None,
    ) -> Union[DictReplayBufferSamples, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]]:
        """
        :param batch_size: Number of element to sample (only used for online sampling)
        :param env: associated gym VecEnv to normalize the observations/rewards
            Only valid when using online sampling
        :param online_sampling: Using online_sampling for HER or not.
        :param n_sampled_goal: Number of sampled goals for replay. (offline sampling)
        :return: Samples.
        """
        cell_idxs = np.random.randint(0, self.nb_cells, batch_size)
        transitions_indices = []
        episode_indices = []
        for cell_idx in cell_idxs:
            cell = self._idx_to_cell[cell_idx]
            episode_idx, transition_idx = self._cell_to_transition(cell)
            episode_indices.append(episode_idx)
            transitions_indices.append(transition_idx)
        episode_indices = np.array(episode_indices)
        transitions_indices = np.array(transitions_indices)

        # Select which episodes to use
        her_indices = np.arange(batch_size)[: int(self.her_ratio * batch_size)]

        # get selected transitions
        transitions = {key: self._buffer[key][episode_indices, transitions_indices].copy() for key in self._buffer.keys()}

        # sample new desired goals and relabel the transitions
        new_goals = self.sample_goals(episode_indices, her_indices, transitions_indices)
        transitions["desired_goal"][her_indices] = new_goals

        # Convert info buffer to numpy array
        transitions["info"] = np.array(
            [
                self.info_buffer[episode_idx][transition_idx]
                for episode_idx, transition_idx in zip(episode_indices, transitions_indices)
            ]
        )

        # Edge case: episode of one timesteps with the future strategy
        # no virtual transition can be created
        if len(her_indices) > 0:
            # Vectorized computation of the new reward
            transitions["reward"][her_indices, 0] = self.env.env_method(
                "compute_reward",
                # the new state depends on the previous state and action
                # s_{t+1} = f(s_t, a_t)
                # so the next_achieved_goal depends also on the previous state and action
                # because we are in a GoalEnv:
                # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
                # therefore we have to use "next_achieved_goal" and not "achieved_goal"
                transitions["next_achieved_goal"][her_indices, 0],
                # here we use the new desired goal
                transitions["desired_goal"][her_indices, 0],
                transitions["info"][her_indices, 0],
            )

        # concatenate observation with (desired) goal
        observations = self._normalize_obs(transitions, maybe_vec_env)

        # HACK to make normalize obs and `add()` work with the next observation
        next_observations = {
            "observation": transitions["next_obs"],
            "achieved_goal": transitions["next_achieved_goal"],
            # The desired goal for the next observation must be the same as the previous one
            "desired_goal": transitions["desired_goal"],
        }
        next_observations = self._normalize_obs(next_observations, maybe_vec_env)

        if online_sampling:
            next_obs = {key: self.to_torch(next_observations[key][:, 0, :]) for key in self._observation_keys}

            normalized_obs = {key: self.to_torch(observations[key][:, 0, :]) for key in self._observation_keys}

            return DictReplayBufferSamples(
                observations=normalized_obs,
                actions=self.to_torch(transitions["action"]),
                next_observations=next_obs,
                dones=self.to_torch(transitions["done"]),
                rewards=self.to_torch(self._normalize_reward(transitions["reward"], maybe_vec_env)),
            )
        else:
            return observations, next_observations, transitions["action"], transitions["reward"]
