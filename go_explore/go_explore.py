import copy
from typing import Any, Dict, Mapping, Optional, Tuple

import gym
import numpy as np
import torch as th
from gym import Env, spaces

from go_explore.archive import ArchiveBuffer
from go_explore.cells import CellFactory


class Goalify(gym.GoalEnv, gym.Wrapper):
    """
    Wrap the env into a GoalEnv.

    :param env: The environment
    :param cell_factory: The cell factory
    :param nb_random_exploration_steps: Number of random exploration steps after the goal is reached, defaults to 30
    :param window_size: Agent can skip goals in the goal trajectory within the limit of ``window_size``
        goals ahead, defaults to 10
    """

    def __init__(
        self,
        env: Env,
        cell_factory: CellFactory,
        nb_random_exploration_steps: int = 30,
        window_size: int = 10,
    ) -> None:
        super().__init__(env)
        # Set a goal-conditionned observation space
        self.observation_space = spaces.Dict(
            {
                "observation": copy.deepcopy(self.env.observation_space),
                "desired_goal": copy.deepcopy(self.env.observation_space),
                "achieved_goal": copy.deepcopy(self.env.observation_space),
            }
        )
        self.archive = None  # type: ArchiveBuffer

        self.cell_factory = cell_factory
        self.nb_random_exploration_steps = nb_random_exploration_steps
        self.window_size = window_size

    def set_archive(self, archive: ArchiveBuffer) -> None:
        """
        Set the archive.

        :param archive: The archive
        """
        self.archive = archive

    def reset(self) -> Dict[str, np.ndarray]:
        obs = self.env.reset()
        assert self.archive is not None, "you need to set the archive before reset. Use set_archive()"
        # TODO: change here
        self.goal_trajectory = [self.observation_space["desired_goal"].sample() for _ in range(3)]
        # self.goal_trajectory = self.archive.sample_goal_trajectory()
        self._goal_idx = 0
        self.done_countdown = self.nb_random_exploration_steps
        self._is_last_goal_reached = False  # useful flag
        obs = self._get_obs(obs)  # turn into dict
        return obs

    def _get_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "observation": obs.copy(),
            "achieved_goal": obs.copy(),
            "desired_goal": self.goal_trajectory[self._goal_idx].copy(),
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        wrapped_obs, reward, done, info = self.env.step(action)

        # Compute reward (has to be done before moving to next goal)
        desired_goal = self.goal_trajectory[self._goal_idx]
        reward = self.compute_reward(wrapped_obs, desired_goal, {})

        # Move to next goal here (by modifying self._goal_idx and self._is_last_goal_reached)
        self.maybe_move_to_next_goal(wrapped_obs)

        # When the last goal is reached, delay the done to allow some random actions
        if self._is_last_goal_reached:
            if self.done_countdown != 0:
                info["use_random_action"] = True
                self.done_countdown -= 1
            else:  # self.done_countdown == 0:
                done = True

        obs = self._get_obs(wrapped_obs)
        return obs, reward, done, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Mapping[str, Any]) -> float:
        if len(achieved_goal.shape) == len(self.observation_space["achieved_goal"].shape):
            is_success = self.is_success(achieved_goal, desired_goal)
        elif len(achieved_goal.shape) == len(self.observation_space["achieved_goal"].shape) + 1:
            # Here, the samples comes from the buffer. In SB3, when observation are images
            # the buffer store transposed images. That's why we need to transpose this images
            # before computing success. Remove the following 2 lines if obs are not images.
            achieved_goal = np.moveaxis(achieved_goal, 1, -1)
            desired_goal = np.moveaxis(desired_goal, 1, -1)
            is_success = self.is_success(achieved_goal, desired_goal, dim=0)
        else:
            raise ValueError("dim can only be either None or 0")
        return is_success - 1

    def is_success(self, obs: np.ndarray, goal: np.ndarray, dim: Optional[int] = None) -> np.ndarray:
        """
        Return True when the obs and the goal are in the same cell.

        :param obs: The observation
        :param goal: The goal
        :return: Success or not
        """
        cells = self.cell_factory(th.from_numpy(obs))
        goal_cells = self.cell_factory(th.from_numpy(goal))
        if dim is None:
            return (cells == goal_cells).all().cpu().numpy()
        elif dim == 0:
            return np.array([(cell == goal_cell).all() for cell, goal_cell in zip(cells, goal_cells)])
        else:
            raise ValueError("dim can only be either None or 0")

    def maybe_move_to_next_goal(self, obs: np.ndarray) -> None:
        """
        Set the next goal idx if necessary.

        From the paper:
        "When a cell that was reached occurs multiple times in the window, the next goal
        is the one that follows the last occurence of this repeated goal cell."

        :param obs: The observation
        """
        upper_idx = min(self._goal_idx + self.window_size, len(self.goal_trajectory))
        for goal_idx in range(self._goal_idx, upper_idx):
            goal = self.goal_trajectory[goal_idx]
            if self.is_success(obs, goal):
                self._goal_idx = goal_idx + 1
        # Update the flag _is_last_goal_reached
        if self._goal_idx == len(self.goal_trajectory):
            self._is_last_goal_reached = True
            self._goal_idx -= 1


class GoExplore:
    pass
