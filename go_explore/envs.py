import copy
from typing import Any, Dict, List, Tuple, Union

import gym
import numpy as np
import panda_gym
from gym import register, spaces

from go_explore.common.wrappers import EpisodeStartWrapper, UnGoalWrapper
from go_explore.go_explore.archive import ArchiveBuffer
from go_explore.go_explore.cell_computers import CellComputer


class _ContinuousMinigrid(gym.Env):
    """
    Simple small gridworld with continuous spaces. Used to test algorithms.

    No, goal, so no reward.
    """

    def __init__(self) -> None:
        self.observation_space = spaces.Box(-20, 20, (2,))
        self.action_space = spaces.Box(-1, 1, (2,))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # if        action < -1/2 , action = -1/2
        # if  1/2 < action        , action =  1/2
        # if -1/2 < action <  1/2 , action =    0
        action = np.array(action)
        action = (action > 0.5) * 1.0 + (action < -0.5) * -1.0
        self.pos = np.clip(self.pos + action, -10, 10)
        return np.copy(self.pos), 0.0, False, {}

    def reset(self) -> Dict[str, np.ndarray]:
        self.pos = np.array([0.0, 0.0])
        return np.copy(self.pos)


def ContinuousMinigrid():
    env = _ContinuousMinigrid()
    env = EpisodeStartWrapper(env)  # needed to store properly in archive
    return env


register(
    id="ContinuousMinigrid-v0",
    entry_point="go_explore.envs:ContinuousMinigrid",
    max_episode_steps=10,
)


class SubgoalEnv(gym.GoalEnv):
    """
    Panda environment with subgoal.

    Reward is 0.0 when reached observation and desired observation share the same cell. -1.0 otehrwise.
    Inheritance should implement generate_subgoals.

    :param env: the environment
    :param cell_computer: the cell computer
    :param subgoal_horizon: the subgoal horizon, defaults to 1
    :param done_delay: number of random action after the goal is reached, defaults to 0
    :param count_pow: count pow when sampling goal, defaults to 0
    """

    def __init__(
        self, env: gym.Env, cell_computer: CellComputer, subgoal_horizon: int = 1, done_delay: int = 0, count_pow: int = 0
    ) -> None:

        self.env = EpisodeStartWrapper(env)
        self.observation_space = spaces.Dict(
            {
                "observation": copy.deepcopy(self.env.observation_space),
                "desired_goal": copy.deepcopy(self.env.observation_space),
                "achieved_goal": copy.deepcopy(self.env.observation_space),
            }
        )
        self.action_space = self.env.action_space
        self.done_delay = done_delay
        self.subgoal_horizon = subgoal_horizon
        self.cell_computer = cell_computer
        self.archive = ArchiveBuffer(
            1_000_000, self.env.observation_space, self.env.action_space, self.cell_computer, count_pow
        )
        self.subgoal_idx = 0

    @property
    def spec(self):
        return self.env.spec

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        wrapped_obs, reward, done, info = self.env.step(action)
        observation = np.copy(wrapped_obs)
        achieved_goal = np.copy(wrapped_obs)
        desired_goal = np.copy(self.subgoals[self.subgoal_idx])
        reward = self.compute_reward(achieved_goal, desired_goal, {})
        if reward == 0.0:
            if self.subgoal_idx + 1 == len(self.subgoals):  # last goal reached
                self.is_success = True
            else:
                self.subgoal_idx += 1
        obs = {
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
            "observation": observation,
        }
        if self.is_success and self.done_countdown == 0:
            done = True
            info["is_success"] = 1.0
        elif self.is_success and self.done_countdown != 0:
            info["done"] = True
            info["is_success"] = 1.0
            self.done_countdown -= 1
        else:
            info["is_success"] = 0.0
        return obs, reward, done, info

    def reset(self) -> Dict[str, np.ndarray]:
        obs = self.env.reset()
        self.subgoals = self.generate_subgoals(obs, self.subgoal_horizon)
        self.subgoal_idx = 0
        achieved_goal = np.copy(obs)
        desired_goal = np.copy(self.subgoals[self.subgoal_idx])
        self.is_success = False
        self.done_countdown = self.done_delay
        obs = {
            "observation": obs,
            "desired_goal": desired_goal,
            "achieved_goal": achieved_goal,
        }
        return obs

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any]
    ) -> Union[float, np.ndarray]:
        """
        Returns 0.0 if achieved_goal and desired_goal share the same cell, -1.0 otherwise.

        This method is vectorized, which means that if you pass an array of desired and
        achieved goals as input, the method returns an array of rewards.

        :param achieved_goal: Achieved goal or array of achieved goals
        :param desired_goal: Desired goal or array of desired goals
        :param info: Unused, for consistency
        :return: Reward or array of rewards, depending on the input
        """
        if achieved_goal.shape == self.observation_space["achieved_goal"].shape:  # single obs
            achieved_goal_cell = self.cell_computer.compute_cell(achieved_goal)
            desired_goal_cell = self.cell_computer.compute_cell(desired_goal)
            if achieved_goal_cell == desired_goal_cell:
                return 0.0
            else:
                return -1.0
        elif achieved_goal.shape[1:] == self.observation_space["achieved_goal"].shape:  # multiple obs
            achieved_goal_cells = self.cell_computer.compute_cells(achieved_goal)  # faster than recursive
            desired_goal_cells = self.cell_computer.compute_cells(desired_goal)
            are_egals = [
                achieved_goal_cell == desired_goal_cell
                for (achieved_goal_cell, desired_goal_cell) in zip(achieved_goal_cells, desired_goal_cells)
            ]
            rewards = np.array(are_egals, dtype=np.float32) - 1.0
            return rewards
        else:
            raise ValueError(
                "desired_goal and achieved_goal should be either elements contained \
                in the observation space or an array of elements contained in the observation space."
            )

    def generate_subgoals(self, obs: np.ndarray, subgoal_horizon: int = 1) -> List[np.ndarray]:
        """
        Sample a subgoal path from the archive.

        :param obs: the current observation
        :param subgoal_horizon: the subgoal horizon, defaults to 1
        :return: the subgoal path
        """
        try:
            # Sometimes the next line raises an exception, which means that the current cell is not
            # contained in the archive. This can happen at the very beginning, when the archive is
            # still empty, or if this method is called when the agent has just discovered a new cell.
            goals = self.archive.sample_subgoal_path(obs, subgoal_horizon)
        except KeyError:
            goals = [np.zeros(self.observation_space["observation"].shape)]
        return goals


def PandaReachFlat(**kwargs):
    env = gym.make("PandaReach-v2", **kwargs)
    env = UnGoalWrapper(env)
    return env
