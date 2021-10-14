import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import gym.spaces
import numpy as np
from gym.wrappers.time_limit import TimeLimit

from go_explore.buffer import PathfinderBuffer
from go_explore.cell_computers import CellComputer


class IntermediateGoalEnv(gym.GoalEnv, ABC):
    @abstractmethod
    def hard_reset(self) -> None:
        """
        True reset of the environment.
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Soft reset, the state remain the same, but the goal changes.

        :return: The observation, the desired goal and the achived goal.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_goal(self, goal: np.ndarray) -> None:
        """
        Set the goal. Must be called before reset.

        :param goal: The new goal
        """
        raise NotImplementedError()


class IntermediateGoalWrapper(gym.Wrapper):
    """
    Convenient class to wrap IntermediateGoalEnv.

    :param env: The environment to be wrapped
    """

    def __init__(self, env: IntermediateGoalEnv) -> None:
        super().__init__(env)

    def hard_reset(self) -> None:
        """
        True reset of the environment.
        """
        self.env.hard_reset()

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Soft reset, the state remain the same, but the goal changes.

        :return: The observation, the desired goal and the achived goal.
        """
        return super().reset()

    def set_goal(self, goal: np.ndarray) -> None:
        """
        Set the goal. Must be called before reset.

        :param goal: The new goal
        """
        self.env.set_goal(goal)


class GoExploreWrapper(gym.Wrapper, IntermediateGoalEnv):
    """
    Turn the environment into a GoExplore environment.

    The goal is to reach a given observation, contained in ``desired_goal``. The ``achieved_goal`` and the
    ``observation`` are equal. The agent gets a 0.0 reward when the goal is reached, and -1.0 otherwise.
    You can set the goal with ``env.set_goal(goal)``.

    :param env: The envrionment to be wrapped
    :param cell_computer: The cell computer, used to compute the reward
    """

    def __init__(self, env: gym.Env, cell_computer: CellComputer) -> None:
        gym.Wrapper.__init__(self, env=env)
        IntermediateGoalEnv.__init__(self)
        self.cell_computer = cell_computer
        self.observation_space = gym.spaces.Dict(
            {
                "observation": copy.deepcopy(self.env.observation_space),
                "achieved_goal": copy.deepcopy(self.env.observation_space),
                "desired_goal": copy.deepcopy(self.env.observation_space),
            }
        )
        self.goal = self.observation_space["desired_goal"].sample()  # changed later

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        wrapped_obs, wrapped_reward, wrapped_done, wrapped_info = super().step(action)
        self.last_obs = wrapped_obs
        obs = self.observation(wrapped_obs)
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], wrapped_info))
        done = False
        info = wrapped_info
        if reward == 0.0:
            info["is_success"] = 1.0
        else:
            info["is_success"] = 0.0
        return obs, reward, done, info

    def reset(self) -> Dict[str, np.ndarray]:
        wrapped_obs = self.last_obs
        obs = self.observation(wrapped_obs)
        return obs

    def hard_reset(self) -> None:
        self.last_obs = self.env.reset()

    def set_goal(self, goal: np.ndarray) -> None:
        self.goal = goal.copy()

    def observation(self, wrapped_obs: np.ndarray) -> Dict[str, np.ndarray]:
        achieved_goal = wrapped_obs.copy()
        desired_goal = self.goal.copy()
        observation = wrapped_obs.copy()
        obs = {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }
        return obs

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any]
    ) -> Union[float, np.ndarray]:
        if achieved_goal.shape == self.observation_space["achieved_goal"].shape:  # single obs
            achieved_goal_cell = self.cell_computer.compute_cell(achieved_goal)
            desired_goal_cell = self.cell_computer.compute_cell(desired_goal)
            if achieved_goal_cell == desired_goal_cell:
                return 0.0
            else:
                return -1.0
        elif achieved_goal.shape[1:] == self.observation_space["achieved_goal"].shape:  # multiple obs
            achieved_goal_cells = self.cell_computer.compute_cells(achieved_goal)
            desired_goal_cells = self.cell_computer.compute_cells(desired_goal)
            are_egals = [
                achieved_goal_cell == desired_goal_cell
                for (achieved_goal_cell, desired_goal_cell) in zip(achieved_goal_cells, desired_goal_cells)
            ]
            rewards = np.array(are_egals, dtype=np.float32) - 1.0
            return rewards
        else:
            raise ValueError


class SparseMoutainCarWrapper(IntermediateGoalWrapper, gym.RewardWrapper):
    """
    The agent gets a 0.0 reward when the goal is reached, -1.0 otherwise.
    """

    def reward(self, reward: float) -> float:
        if reward == 100.0:
            return 0.0
        else:
            return -1.0


class UnGoalWrapper(gym.ObservationWrapper):
    """
    Observation wrapper that flattens the observation. ``achieved_goal`` is removed.

    :param env: The environment to be wrapped
    """

    def __init__(self, env: gym.GoalEnv) -> None:
        super(UnGoalWrapper, self).__init__(env)
        env.observation_space.spaces.pop("achieved_goal")
        self.observation_space = gym.spaces.flatten_space(env.observation_space)

    def observation(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        observation.pop("achieved_goal")
        observation = gym.spaces.flatten(self.env.observation_space, observation)
        return observation

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any]) -> None:
        raise NotImplementedError("This method is not accessible, since the environment is not a GoalEnv.")


class GoalTrajcetorySetterWrapper(IntermediateGoalWrapper, ABC):
    """
    Set a new goal whenever it's needed.

    When `reset` is called,
    - if the current goal is reached, it sets the next goal,
    - if the current goal is not reached, it samples a new trajectory, and set the first goal.

    When `step` is called,
    - if the current goal is reached, info contains {"is_success": 1.0}
    - if the current goal is reached and it was the last one of the trajectory, info contains {"traj_success": 1.0}
    - if done, but the current goal is not reached, info contains {"traj_success": 0.0}.

    :param env: The environment to be wrapped
    """

    def __init__(self, env: IntermediateGoalEnv) -> None:
        super().__init__(env)
        self.need_new_traj = True

    def hard_reset(self) -> None:
        super().hard_reset()
        self.need_new_traj = True

    def reset(self) -> Dict[str, np.ndarray]:
        obs = super().reset()
        if self.need_new_traj:
            try:
                self.goal_trajectory = self.sample_trajectory(obs["observation"])
            except KeyError:  # observation never seen # TODO: is it the best way ?
                self.goal_trajectory = [self.observation_space["desired_goal"].sample()]
            if len(self.goal_trajectory) == 0:  # no obs seen after
                self.goal_trajectory = [self.observation_space["desired_goal"].sample()]
            self.goal_idx = 0
            self.need_new_traj = False
        goal = self.goal_trajectory[self.goal_idx]
        self.set_goal(goal)
        obs["desired_goal"] = self.goal.copy()
        return obs

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        obs, reward, done, info = super().step(action)
        is_success = reward == 0.0
        if is_success:
            done = True
            self.goal_idx += 1
            if self.goal_idx == len(self.goal_trajectory):  # last goal rof trajctory reached
                self.need_new_traj = True
                info["traj_success"] = 1.0
        elif done:  # done because goal not reached in time, for instance
            self.need_new_traj = True
            info["traj_success"] = 0.0
        return obs, reward, done, info

    @abstractmethod
    def sample_trajectory(self, from_obs: np.ndarray) -> List[np.ndarray]:
        """
        Sample a trajectory of goals.

        :param from_obs: The observation taken as a starting point
        :return: A trajectory of observations
        """
        raise NotImplementedError()


class HardResetAfterTrajWrapper(IntermediateGoalWrapper):
    """
    Hard reset the environment when a trajcetory is done (failure and success).

    You do not need to handle hard_reset once your environment is wrapped.

    :param env: The environment to be wrapped
    """

    def __init__(self, env: IntermediateGoalEnv) -> None:
        super().__init__(env)
        self.need_hard_reset = True

    def reset(self) -> Dict[str, np.ndarray]:
        if self.need_hard_reset:
            self.hard_reset()
            self.need_hard_reset = False
        return super().reset()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        obs, reward, done, info = super().step(action)
        if info.get("traj_success") is not None:
            self.need_hard_reset = True
        return obs, reward, done, info


class HardResetSometimesWrapper(IntermediateGoalWrapper):
    """
    Hard reset the environment when a trajcetory is done (failure and success).

    You do not need to handle hard_reset once your environment is wrapped.

    :param env: The environment to be wrapped
    """

    def __init__(self, env: IntermediateGoalEnv, nb_timesteps: Optional[int] = None) -> None:
        super().__init__(env)
        self.nb_timesteps = nb_timesteps
        self._count = 0
        self.need_hard_reset = True

    def reset(self) -> Dict[str, np.ndarray]:
        if self.need_hard_reset:
            self.hard_reset()
            self.need_hard_reset = False
        return super().reset()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        obs, reward, done, info = super().step(action)
        self._count += 1
        if self.nb_timesteps is not None and self._count % self.nb_timesteps == 0:
            self.need_hard_reset = True
        return obs, reward, done, info


class PandaFileGoalTrajcetorySetterWrapper(GoalTrajcetorySetterWrapper):
    """
    Sample a trajectory of 5 goals among the file ``goal.npy``

    :param env: The environment to be wrapped
    """

    def __init__(self, env: IntermediateGoalEnv) -> None:
        super().__init__(env)
        with open("goals.npy", "rb") as f:
            self.goals = np.load(f)
        self.rng = np.random.default_rng()

    def sample_trajectory(self) -> np.ndarray:
        goal_trajectory = self.rng.choice(self.goals, 5)
        return goal_trajectory


class PandaGoalBufferFileTrajcetorySetterWrapper(GoalTrajcetorySetterWrapper):
    """
    Sample a trajectory of 5 goals among the file ``goal.npy``

    :param env: The environment to be wrapped
    """

    def __init__(self, env: IntermediateGoalEnv) -> None:
        super().__init__(env)
        self.fill_buffer()

    def fill_buffer(self) -> None:
        """
        Create an envrionment, and fill the buffer using random actions.
        """
        from go_explore.buffer import PathfinderBuffer
        from go_explore.cell_computers import PandaCellComputer

        env = gym.make("PandaReach-v1")
        env = UnGoalWrapper(env)
        cell_computer = PandaCellComputer()
        env = GoExploreWrapper(env, cell_computer)
        env = TimeLimit(env, 100)

        self.goal_buffer = PathfinderBuffer(100000, env.observation_space, env.action_space, cell_computer)

        for i in range(1000):
            print(1000 - i)
            env.hard_reset()
            obs = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                next_obs, reward, done, info = env.step(action)
                self.goal_buffer.add(obs, next_obs, action, reward, done, [info])
                obs = next_obs

    def sample_trajectory(self, from_obs: np.ndarray) -> List[np.ndarray]:
        goal_trajectory = self.goal_buffer.sample_trajectory(from_obs)
        print("length of traj", len(goal_trajectory))
        return goal_trajectory


class GoalBufferTrajcetorySetterWrapper(GoalTrajcetorySetterWrapper):
    """
    Sample a trajectory of 5 goals among the file ``goal.npy``

    :param env: The environment to be wrapped
    """

    def __init__(self, env: IntermediateGoalEnv, goal_buffer: PathfinderBuffer) -> None:
        super().__init__(env)
        self.goal_buffer = goal_buffer

    def sample_trajectory(self, from_obs: np.ndarray) -> List[np.ndarray]:
        goal_trajectory = self.goal_buffer.sample_trajectory(from_obs)
        return goal_trajectory
