from typing import List, Union

import numpy as np
import torch as th
from gym import Env, spaces
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

from go_explore.cells import CellFactory


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
        assert n_envs == 1, "The trajectory manager is not compatible with multiprocessing"
        self.count_pow = count_pow
        self.cell_factory = cell_factory

    def update_cells(self) -> None:
        """
        Compute the cells and update the trajectories. Must be called after modification of cell_factory parameters.
        """
        self._compute_cells()
        self._update_trajectories()

    def _compute_cells(self) -> None:
        """
        Compute the cells.
        """
        upper_bound = self.pos if not self.full else self.buffer_size
        observations = self.to_torch(self.observations["observation"][:upper_bound])
        self.cells = self.cell_factory(observations)

    def _update_trajectories(self) -> None:
        """
        Update the trajectories based on the cells. Must be called after updating cells.
        """
        upper_bound = self.pos if not self.full else self.buffer_size
        nb_obs = upper_bound * self.n_envs
        flat_cells = self.cells.reshape((nb_obs, -1))  # shape from (pos, env_idx, *cell_shape) to (idx, *cell_shape)
        # Compute the unique cells.
        # cells_uid is a tensor of shape (nb_obs,) mapping observation index to its cell index.
        # unique_cells is a tensor of shape (nb_cells, *cell_shape) mapping cell index to the cell.
        unique_cells, cells_uid, self.counts = th.unique(flat_cells, return_inverse=True, return_counts=True, dim=0)
        self.nb_cells = unique_cells.shape[0]  # number of unique cells
        flat_pos = th.arange(self.ep_start.shape[0]).repeat_interleave(self.n_envs)  # [0, 0, 1, 1, 2, ...] if n_envs == 2
        flat_ep_start = th.from_numpy(self.ep_start).flatten()  # shape from (pos, env_idx) to (idx,)
        flat_timestep = flat_pos - flat_ep_start  # timestep within the episode

        earliest_cell_occurence = th.zeros(self.nb_cells, dtype=th.int64)
        for cell_uid in range(self.nb_cells):
            cell_idxs = th.where(cell_uid == cells_uid)[0]  # index of observations that are in the cell
            all_cell_occurences_timestep = flat_timestep[cell_idxs]  # the cell has been visited after all these timesteps
            earliest = th.argmin(all_cell_occurences_timestep)  # focus on the time when the cell is visited the earliest
            earliest_cell_occurence[cell_uid] = cell_idxs[earliest]

        # earliest_cell_envs maps cell uid to the env index of the earliest cell visitation
        # earliest_cell_pos maps cell uid to the buffer position of the earliest cell visitation
        # earliest_cell_start maps cell uid to the buffer position ofstarting of the episode when cell has been visited the earliest
        self.earliest_cell_env = earliest_cell_occurence % self.n_envs
        self.earliest_cell_pos = th.div(earliest_cell_occurence, self.n_envs, rounding_mode="floor")
        self.earliest_cell_start = flat_ep_start[earliest_cell_occurence]

    def sample_cell_trajectory(self) -> List[np.ndarray]:
        """
        Sample a trajcetory of cells.

        :return: A list of non-repatitive cells as Tensor
        """
        # Weights depending of the cell visitation count
        weights = 1 / th.sqrt(self.counts + 1)
        cell_uid = th.multinomial(weights, 1)
        # Get the env_idx, the pos in the buffer and the position of the start of the trajectory
        env = self.earliest_cell_env[cell_uid]
        start = self.earliest_cell_start[cell_uid]
        pos = self.earliest_cell_pos[cell_uid]
        # Loop to avoid consecutive repetition
        cell_trajectory = [self.cells[start, env].cpu().numpy()]
        for i in range(start, pos + 1):
            cell = self.cells[i, env].cpu().numpy()
            if (cell != cell_trajectory[-1]).any():
                cell_trajectory.append(cell)
        return cell_trajectory
