from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch as th
from gym import Env, spaces
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

from go_explore.cells import CellFactory


def indexes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Indexes of a in b.

    :param a: Array of shape (...)
    :param b: Array of shape (N x ...)
    :return: Indexes of the occurences of a in b
    """
    if b.shape[0] == 0:
        return np.array([])
    a = a.flatten()
    b = b.reshape((b.shape[0], -1))
    idxs = np.where((a == b).all(1))[0]
    return idxs


def index(a: np.ndarray, b: np.ndarray) -> Optional[int]:
    """
    Index of first occurence of a in b.

    :param a: Array of shape (...)
    :param b: Array of shape (N x ...)
    :return: index of the first occurence of a in b
    """
    idxs = indexes(a, b)
    if idxs.shape[0] == 0:
        return None
    else:
        return idxs[0]


def multinomial(weights: np.ndarray) -> int:
    p = weights / weights.sum()
    r = np.random.multinomial(1, p, size=1)
    idx = np.nonzero(r)[1][0]
    return idx


class ArchiveBuffer(HerReplayBuffer):
    """
    HER replay buffer that keep track of cells and cell trajectories.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param env: The envrionment
    :param cell_factory: The cell factory
    :param count_pow: The cell weight is 1 / count**count_pow
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
        env: Env,
        cell_factory: CellFactory,
        count_pow: float = 0,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        n_sampled_goal: int = 4,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
        online_sampling: bool = True,
    ) -> None:
        assert online_sampling, "Not compatible with offline sampling for the moment."
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            env,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy=goal_selection_strategy,
            online_sampling=online_sampling,
        )
        self.counts = np.empty((0,), dtype=np.int64)
        self.earliest_cell_env = np.empty((0,), dtype=np.int64)
        self.earliest_cell_pos = np.empty((0,), dtype=np.int64)
        self.count_pow = count_pow

        self.set_new_cell_factory(cell_factory)

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        th_obs = self.to_torch(next_obs["observation"])
        cells = self.cell_factory(th_obs).cpu().numpy()
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
                current_distance_to_cell = self.pos - self.ep_start[self.pos, env_idx]
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

        return super().add(obs, next_obs, action, reward, done, infos)

    def set_new_cell_factory(self, cell_factory: CellFactory):
        self.cell_factory = cell_factory

        cell_shape = self.cell_factory.cell_space.shape
        self.cells = np.zeros((self.buffer_size, self.n_envs, *cell_shape))
        self.unique_cells = np.empty((0, *cell_shape), dtype=self.cell_factory.cell_space.dtype)
        self._recompute_cells()
        self._update_trajectories()

    def _recompute_cells(self) -> None:
        """
        Compute the cells.
        """
        upper_bound = self.pos if not self.full else self.buffer_size
        # Recompute 256 by 256 to avoid cuda space allocation error.
        k = 0
        while k < upper_bound:
            upper = min(upper_bound, k + 256)
            observations = self.to_torch(self.next_observations["observation"][k:upper])
            self.cells[k:upper] = self.cell_factory(observations).cpu().numpy()
            k += 256

    def _update_trajectories(self) -> None:
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
        self.unique_cells, cells_uid, counts = th.unique(flat_cells, return_inverse=True, return_counts=True, dim=0)
        self.counts = counts.cpu().numpy()  # type: np.ndarray
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
        # earliest_cell_envs maps cell uid to the env index of the earliest cell visitation
        # earliest_cell_pos maps cell uid to the buffer position of the earliest cell visitation
        self.earliest_cell_env = (earliest_cell_occurence % self.n_envs).cpu().numpy()
        self.earliest_cell_pos = th.div(earliest_cell_occurence, self.n_envs, rounding_mode="floor").cpu().numpy()

    def sample_cell_trajectory(self) -> List[np.ndarray]:
        """
        Sample a trajcetory of cells.

        :return: A list of non-repatitive cells as Tensor
        """
        if self.counts.shape[0] == 0:  # no cells yet
            return [self.observation_space["desired_goal"].sample().astype(np.uint8)]
        # Weights depending of the cell visitation count
        weights = 1 / np.sqrt(self.counts + 1)
        cell_uid = multinomial(weights)
        # Get the env_idx, the pos in the buffer and the position of the start of the trajectory
        env = self.earliest_cell_env[cell_uid]
        pos = self.earliest_cell_pos[cell_uid]
        start = self.ep_start[pos, env]
        # Loop to avoid consecutive repetition
        cell_trajectory = [self.cells[start, env]]
        for i in range(start, pos + 1):
            cell = self.cells[i, env]
            if (cell != cell_trajectory[-1]).any():
                cell_trajectory.append(cell)
        return cell_trajectory
