import random
from typing import Any, Callable, Dict, List, Union

import numpy as np
import torch as th
from gym import spaces
from scipy.sparse.csgraph import shortest_path
from stable_baselines3.common.buffers import DictReplayBuffer

from go_explore.cell_computers import Cell, CellComputer


class PathfinderBuffer(DictReplayBuffer):
    """
    Like trajectory buffer but also add a trajectory sampler.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param cell_computer: The cell computer
    :param goal_horizon: Number of cells separating two observations in the goal trajectory
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
        goal_horizon: int = 1,
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
        self.goal_horizon = goal_horizon
        self._cell_to_idx = {}  # A dict mapping cell to a unique idx
        self._idx_to_cell = []  # Same, but the other way. Faster than using .index()
        self._cell_to_obss = {}  # A dict mapping cell to a list of every encountered observation in that cell
        self.nb_cells = 0  # The number of encountered cells
        self.csgraph = np.zeros(
            shape=(0, 0)
        )  # A matrix to store idx that are neighboors (csgraph[5][2] == 1 means that 5 can lead to 2)
        self._counts = np.zeros(shape=(0,))  # A counter of the number of visits per cell
        self.count_pow = count_pow

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        super().add(obs, next_obs, action, reward, done, infos)
        current_cell = self.cell_computer.compute_cell(obs["achieved_goal"])
        next_cell = self.cell_computer.compute_cell(next_obs["achieved_goal"])
        for cell in [current_cell, next_cell]:
            try:
                idx = self._cell_to_idx[cell]
            except KeyError:  # = if cell is visited for the first time:
                idx = len(self._cell_to_idx)
                self._idx_to_cell.append(cell)
                self._cell_to_idx[cell] = idx
                self._cell_to_obss[cell] = []
                self.nb_cells += 1
                # expanding arrays
                self._counts = np.pad(self._counts, (0, 1), constant_values=0)
                self.csgraph = np.pad(self.csgraph, ((0, 1), (0, 1)), constant_values=np.inf)
            # update counts and obs list
            self._counts[idx] += 1
            self._cell_to_obss[cell].append(obs["achieved_goal"])
        # update csgraph
        current_cell_idx = self._cell_to_idx[current_cell]
        next_cell_idx = self._cell_to_idx[next_cell]
        self.csgraph[current_cell_idx][next_cell_idx] = min(self.csgraph[current_cell_idx][next_cell_idx], 1)

    def sample_trajectory(self, from_obs: np.ndarray) -> List[np.ndarray]:
        """
        Samples a trajectory that starts from the given observation.

        First, compute all reachable observations from the given observation (based on the trajectories already encountered).
        Second, samples a final observation from these reachable observations. The less a final observation has been visited,
        the more likely it is to be sampled.
        Third, uses a shortest path algorithm to select intermediate observations for reaching this final observation.

        :param from_obs: The observation taken as a starting point
        :return: The trajectory of observations.
        """
        # compute the initial cell
        from_cell = self.cell_computer.compute_cell(from_obs)
        from_idx = self._cell_to_idx[from_cell]
        # process the dist matrix, to determine the reachable cells.
        dist_matrix, predecessors = shortest_path(self.csgraph, return_predecessors=True)
        reachable_idxs = self._get_reachable_idxs(from_idx, dist_matrix)
        if len(reachable_idxs) == 0:  # if there is no reacheable cells
            return []
        # compute the weights associated with these reachable cells.
        weights = self._get_weights(reachable_idxs)
        p = weights / weights.sum()
        # randomly choose a final cell
        to_idx = np.random.choice(reachable_idxs, p=p)
        idx_trajectory = self._get_path(predecessors, from_idx, to_idx)
        idx_trajectory.pop(0)  # no need to take the current cell
        # convert cells into observations
        cell_trajectory = [self._idx_to_cell[idx] for idx in idx_trajectory]
        obs_trajectory = [self._cell_to_obs(cell) for cell in cell_trajectory]
        obs_trajectory = self._lighten_trajectory(obs_trajectory)
        return obs_trajectory

    def plan_trajectory(self, from_obs: np.ndarray, compute_success: Callable[[np.ndarray], np.ndarray]) -> List[np.ndarray]:
        """
        Plan the quikest trajectory toward the given goal.

        :param from_obs: The observation taken as a starting point
        :param compute_success: The function used to determine whether an observation is a success or not
        :return: A trajctory of goals.
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
        is_success = compute_success(reachable_obs).squeeze()
        # only consider the observations, distance that correspond to a success
        distances = distances[is_success]
        reachable_idxs = reachable_idxs[is_success]
        # take the min distance, and get the target_obs
        idx = np.argmin(distances)
        to_idx = reachable_idxs[idx]
        # compute the path toward the target_obs and return
        idx_trajectory = self._get_path(predecessors, from_idx, to_idx)
        # convert cells into observations
        cell_trajectory = [self._idx_to_cell[idx] for idx in idx_trajectory]
        obs_trajectory = [self._cell_to_obs(cell) for cell in cell_trajectory]
        obs_trajectory = self._lighten_trajectory(obs_trajectory)
        return obs_trajectory

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
        weights = 1 / (self._counts ** self.count_pow * np.sqrt(1 + self._counts))
        # weights = 1 / (np.sqrt(1 + self._counts))
        # weights = np.ones_like(self._counts)
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

    def _lighten_trajectory(self, trajectory: List[np.ndarray]) -> List[np.ndarray]:
        """
        Pick a list of elements from the list, evenly spaced by a certain number of steps, keeping the last one.

        Example:
        >>> lighten_trajectory([1, 2, 3, 4, 5, 6], step=3)
        [3, 6]

        :param trajectory: The trajectory
        :return: The lightened trajectory
        """
        trajectory = trajectory[:: -self.goal_horizon]
        trajectory.reverse()
        return trajectory
