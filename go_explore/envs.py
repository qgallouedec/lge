import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import gym
import numpy as np
import panda_gym
from gym import register, spaces

from go_explore.common.wrappers import EpisodeStartWrapper, UnGoalWrapper
from go_explore.go_explore.archive import ArchiveBuffer
from go_explore.go_explore.cell_computers import PandaCellComputer, PandaObjectCellComputer


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
        self.pos = self.pos + action
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


class _SubgoalContinuousMinigrid(gym.GoalEnv):
    """
    Simple small gridworld with continuous spaces.

    SubgoalEnv, meaning that goal changes during episode. Done is True when last goal is reached.
    0.0 reward when subgoal reached, -1.0 otherwise.
    """

    def __init__(self) -> None:
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-5, 5, (2,)),
                "desired_goal": spaces.Box(-5, 5, (2,)),
                "achieved_goal": spaces.Box(-5, 5, (2,)),
            }
        )
        self.action_space = spaces.Box(-1, 1, (2,))
        self.subgoal_idx = 0

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        # if        action < -1/2 , action = -1/2
        # if  1/2 < action        , action =  1/2
        # if -1/2 < action <  1/2 , action =    0
        action = np.array(action)
        action = (action > 0.5) * 1.0 + (action < -0.5) * -1.0
        self.pos = np.clip(self.pos + action, -5, 5)
        obs = self.observation()
        reward = self.compute_reward(obs["desired_goal"], obs["achieved_goal"], {})
        if reward == 0.0:
            info, done = {"is_success": 1.0}, False
            self.subgoal_idx += 1
            if self.subgoal_idx == len(self.subgoals):
                done = True
            else:
                obs["desired_goal"] = np.copy(self.subgoals[self.subgoal_idx])
        else:
            info, done = {"is_success": 0.0}, False
        return obs, reward, done, info

    def reset(self) -> Dict[str, np.ndarray]:
        self.pos = np.array([0.0, 0.0])
        self.subgoals = np.random.randint(-5, 5, (2, 2)).astype(np.float32)
        self.subgoal_idx = 0
        return self.observation()

    def observation(self):
        return {
            "observation": np.copy(self.pos),
            "desired_goal": np.copy(self.subgoals[self.subgoal_idx]),
            "achieved_goal": np.copy(self.pos),
        }

    def compute_reward(self, desired_goal, achieved_goal, info):
        reward = (desired_goal == achieved_goal).all(-1).astype(np.float32) - 1.0
        return reward


def SubgoalContinuousMinigrid():
    env = _SubgoalContinuousMinigrid()
    env = EpisodeStartWrapper(env)  # needed to store properly in archive
    return env


register(
    id="SubgoalContinuousMinigrid-v0",
    entry_point="go_explore.envs:SubgoalContinuousMinigrid",
    max_episode_steps=20,
)


class PandaSubgoal(gym.GoalEnv, ABC):
    """
    Panda environment with subgoal.

    Reward is 0.0 when reached observation and desired observation share the same cell. -1.0 otehrwise.
    Inheritance should implement generate_subgoals.
    """

    def __init__(self, done_delay: int = 0, nb_objects: int = 0, render=False) -> None:
        self.env = gym.make("PandaNoTask-v0", nb_objects=nb_objects, render=render)
        self.observation_space = spaces.Dict(
            {
                "observation": copy.deepcopy(self.env.observation_space),
                "desired_goal": copy.deepcopy(self.env.observation_space),
                "achieved_goal": copy.deepcopy(self.env.observation_space),
            }
        )
        self.action_space = self.env.action_space
        if nb_objects == 0:
            self.cell_computer = PandaCellComputer()
        elif nb_objects == 1:
            self.cell_computer = PandaObjectCellComputer()
        self.done_delay = done_delay

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        action = np.array(action)
        self.obs, _, _, info = self.env.step(action)
        desired_goal = self.subgoals[self.subgoal_idx]
        achieved_goal = self.obs
        reward = self.compute_reward(desired_goal, achieved_goal, {})
        if reward == 0.0:
            info = {"is_success": 1.0}
            if self.subgoal_idx + 1 == len(self.subgoals):  # last goal reached
                self.done = True
            else:
                self.subgoal_idx += 1
        else:
            info = {"is_success": 0.0}

        obs = {
            "observation": np.copy(self.obs),
            "desired_goal": np.copy(self.subgoals[self.subgoal_idx]),
            "achieved_goal": np.copy(self.obs),
        }
        if self.done and self.done_countdown == 0:
            done = True
        elif self.done and self.done_countdown != 0:
            done = False
            info["done"] = True
            self.done_countdown -= 1
        else:
            done = False
        return obs, reward, done, info

    def reset(self) -> Dict[str, np.ndarray]:
        self.obs = self.env.reset()
        self.subgoals = self.generate_subgoals()
        self.subgoal_idx = 0
        self.done = False
        self.done_countdown = self.done_delay
        obs = {
            "observation": np.copy(self.obs),
            "desired_goal": np.copy(self.subgoals[self.subgoal_idx]),
            "achieved_goal": np.copy(self.obs),
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

    @abstractmethod
    def generate_subgoals(self) -> List[np.ndarray]:
        """
        Returns a list of subgoal. The agent wants reach the subgoals until the last one.
        """
        ...


class PandaSubgoalRandom(PandaSubgoal):
    def generate_subgoals(self):
        """
        Generate 2 random subgoals.
        """
        goal_range_low = np.array([-0.15, -0.15, 0.0, 0.0, 0.0, 0.0, 0.0])
        goal_range_high = np.array([0.15, 0.15, 0.3, 0.0, 0.0, 0.0, 0.0])
        goals = np.random.uniform(goal_range_low, goal_range_high, size=(2, 7))
        return goals


register(
    id="PandaSubgoalRandom-v0",
    entry_point="go_explore.envs:PandaSubgoalRandom",
    max_episode_steps=50,
)


class _PandaSubgoalArchive(PandaSubgoal):
    """
    Panda Subgoal environment. Subgoals are sample for the archive.

    :param archive: archive from which the subgoals are sampled
    :type archive: ArchiveBuffer
    :param render: whether rendering is enabled, defaults to False
    :type render: bool, optional
    """

    def __init__(self, done_delay: int = 0, nb_objects: int = 0, render: bool = False) -> None:
        super().__init__(done_delay=done_delay, nb_objects=nb_objects, render=render)
        self.archive = ArchiveBuffer(1000000, self.observation_space["observation"], self.action_space, self.cell_computer)

    def generate_subgoals(self) -> List[np.ndarray]:
        """
        Sample a subgoal path from the archive.
        """
        try:
            # Sometimes the next line raises an exception, which means that the current cell is not
            # contained in the archive. This can happen at the very beginning, when the archive is
            # still empty, or if this method is called when the agent has just discovered a new cell.
            goals = self.archive.sample_subgoal_path(self.obs)
        except KeyError:
            goals = [self.env.observation_space.sample()]
        return goals


def PandaSubgoalArchive(**kwargs):
    env = _PandaSubgoalArchive(**kwargs)
    env = EpisodeStartWrapper(env)  # needed to store properly in archive
    return env


register(
    id="PandaSubgoalArchive-v0",
    entry_point="go_explore.envs:PandaSubgoalArchive",
    max_episode_steps=50,
)


def PandaReachFlat(**kwargs):
    env = gym.make("PandaReach-v2", **kwargs)
    env = UnGoalWrapper(env)  # needed to store properly in archive
    return env


register(
    id="PandaReachFlat-v0",
    entry_point="go_explore.envs:PandaReachFlat",
    max_episode_steps=50,
)
