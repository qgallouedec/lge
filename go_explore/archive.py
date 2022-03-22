import random

from typing import Union

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

    def set_cell_factory(self, cell_factory: CellFactory) -> None:
        self.cell_factory = cell_factory

    def update_cells(self):
        upper_bound = self.pos if not self.full else self.buffer_size
        observations = self.to_torch(self.observations["observation"][:upper_bound].squeeze())
        observations = observations.transpose(-1, -3)
        self.cells = self.cell_factory(observations)
