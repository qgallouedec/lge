import copy
from typing import Any, Callable, Dict, Optional, Tuple, Type

import gym
import numpy as np
from gym import Env, spaces
from stable_baselines3.common.base_class import maybe_make_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import is_image_space

from go_explore.archive import ArchiveBuffer
from go_explore.cells import LatentCelling
from go_explore.feature_extractor import GoExploreExtractor


class Goalify(gym.Wrapper):
    """
    Wrap the env into a GoalEnv.

    :param env: The environment
    :param nb_random_exploration_steps: Number of random exploration steps after the goal is reached, defaults to 30
    :param window_size: Agent can skip goals in the goal trajectory within the limit of ``window_size``
        goals ahead, defaults to 10
    """

    def __init__(
        self,
        env: Env,
        nb_random_exploration_steps: int = 30,
        window_size: int = 10,
    ) -> None:
        super().__init__(env)
        # Set a goal-conditionned observation space
        self.observation_space = spaces.Dict(
            {
                "observation": copy.deepcopy(self.env.observation_space),
                "goal": copy.deepcopy(self.env.observation_space),
            }
        )
        self.archive = None  # type: ArchiveBuffer
        self.nb_random_exploration_steps = nb_random_exploration_steps
        self.window_size = window_size

    def set_archive(self, archive: ArchiveBuffer) -> None:
        """
        Set the archive.

        The archive is used to compute goal traejctories, and to compute the cell for the reward.

        :param archive: The archive
        """
        self.archive = archive

    def reset(self) -> Dict[str, np.ndarray]:
        obs = self.env.reset()
        assert self.archive is not None, "you need to set the archive before reset. Use set_archive()"
        self.goal_trajectory = self.archive.sample_trajectory()
        if is_image_space(self.observation_space["goal"]):
            self.goal_trajectory = [goal.transpose(1, 2, 0) for goal in self.goal_trajectory]
        self._goal_idx = 0
        self.done_countdown = self.nb_random_exploration_steps
        self._is_last_goal_reached = False  # useful flag
        dict_obs = self._get_dict_obs(obs)  # turn into dict
        return dict_obs

    def _get_dict_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "observation": obs.copy(),
            "goal": self.goal_trajectory[self._goal_idx].copy(),
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        # Compute reward (has to be done before moving to next goal)
        goal = self.goal_trajectory[self._goal_idx]
        reward = float(self.compute_reward(obs, goal))

        # Move to next goal here (by modifying self._goal_idx and self._is_last_goal_reached)
        self.maybe_move_to_next_goal(obs)

        # When the last goal is reached, delay the done to allow some random actions
        if self._is_last_goal_reached:
            info["is_success"] = True
            if self.done_countdown != 0:
                info["action_repeat"] = action
                self.done_countdown -= 1
            else:  # self.done_countdown == 0:
                done = True
        else:
            info["is_success"] = False

        dict_obs = self._get_dict_obs(obs)
        return dict_obs, reward, done, info

    def compute_reward(self, obs: np.ndarray, goal: np.ndarray, info: Optional[Dict] = None) -> np.ndarray:
        is_success = self.is_success(obs, goal)
        return is_success - 1

    def is_success(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        Return True when the observation and the goal observation are in the same cell.

        :param obs: The observation
        :param goal: The goal observation
        :return: Success or not
        """
        cell = self.archive.compute_cell(obs)
        goal_cell = self.archive.compute_cell(goal)
        return np.isclose(cell, goal_cell, atol=0.001).all(-1)

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


class CallEveryNTimesteps(BaseCallback):
    """
    Callback that calls a function every ``call_freq`` timesteps.

    :param func: The function to call
    :param call_freq: The call timestep frequency, defaults to 1
    :param verbose: Verbosity level 0: not output 1: info 2: debug, defaults to 0
    """

    def __init__(self, func: Callable[[], None], call_freq: int = 1, verbose=0) -> None:
        super(CallEveryNTimesteps, self).__init__(verbose)
        self.func = func
        self.call_freq = call_freq

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.num_timesteps % self.call_freq == 0:
            self.func()

        return True


class GoExplore:
    """
    Go-Explore implementation as described in [1].

    This is a simplified version, which does not include a number of tricks
    whose impact on performance we do not know. The goal is to implement the
    general principle of the algorithm and not all the little tricks.
    In particular, we do not implement:
    - everything related with domain knowledge,
    - self-imitation learning,
    - parallelized exploration phase
    """

    def __init__(
        self,
        model_class: Type[OffPolicyAlgorithm],
        env: Env,
        count_pow: float = 1,
        split_factor: float = 0.125,
        n_envs: int = 1,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
    ) -> None:
        self.split_factor = split_factor
        # Wrap the env
        def env_func():
            return Goalify(maybe_make_env(env, verbose))

        env = make_vec_env(env_func, n_envs=n_envs)
        self.cell_factory = LatentCelling(env.observation_space["observation"])
        replay_buffer_kwargs = {} if replay_buffer_kwargs is None else replay_buffer_kwargs
        replay_buffer_kwargs.update(dict(cell_factory=self.cell_factory, count_pow=count_pow))
        policy_kwargs = dict(features_extractor_class=GoExploreExtractor)
        model_kwargs = {} if model_kwargs is None else model_kwargs

        self.model = model_class(
            "MultiInputPolicy",
            env,
            replay_buffer_class=ArchiveBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            **model_kwargs
        )
        self.archive = self.model.replay_buffer  # type: ArchiveBuffer
        self.archive.set_env(env)
        for _env in self.model.env.envs:
            _env.set_archive(self.archive)

        self.cell_factory.set_feature_extractor(self.model.policy.actor.features_extractor.observation_extractor)

    def _update_cell_factory_param(self) -> None:
        samples = self.archive.sample(512).next_observations["observation"]
        score = self.cell_factory.optimize_param(samples, split_factor=self.split_factor)
        self.model.logger.log("New parameters for cell factory with score", score, "\tStep:", str(self.cell_factory.step))
        self.archive.when_cell_factory_updated()

    def explore(self, total_timesteps: int, update_cell_factory_freq=40000, reset_num_timesteps: bool = False) -> None:
        """
        Run exploration.

        :param total_timesteps: Total timestep of exploration.
        :param reset_num_timesteps: Whether or not to reset the current timestep number (used in logging), defaults to False
        :param update_freq: Cells update frequency
        """
        cell_factory_updater = CallEveryNTimesteps(self._update_cell_factory_param, update_cell_factory_freq)
        self.model.learn(total_timesteps, callback=cell_factory_updater, reset_num_timesteps=reset_num_timesteps)
