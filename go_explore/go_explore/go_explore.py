from typing import Any, Callable, Optional, Tuple, Type

import numpy as np
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.ddpg.ddpg import DDPG
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from go_explore.common.callbacks import LogNbCellsCallback, SaveNbCellsCallback


class GoExplore:
    """
    Go-Explore paradigma, with SAC.

    :param env: The environment to learn from (if registered in Gym, can be str)
    """

    def __init__(self, env) -> None:
        self.explore_model = DDPG("MultiInputPolicy", env, replay_buffer_class=HerReplayBuffer, verbose=1)
        self.explore_model.replay_buffer.child_buffer = env.archive
        self.cell_logger = LogNbCellsCallback(env.archive)
        self.cell_saver = SaveNbCellsCallback(env.archive, save_fq=1000)

    def exploration(self, total_timesteps: int) -> None:
        """
        Exploration phase, switch between go and explore.

        :param total_timesteps: Total number of timesteps
        """
        while self.explore_model.num_timesteps < total_timesteps:
            self.go()
            # self.explore(50)

    def go(self) -> None:
        """
        Go phase. Ends when a trajectory is entirely achieved.

        :param total_timesteps: Max number of timesteps. Learning stops early if a final goal is reached
        """
        print("go")
        callbacks = [self.cell_logger, self.cell_saver]
        self.explore_model.learn(10000, reset_num_timesteps=False, callback=callbacks)

    def explore(self, explore_timesteps: int) -> None:
        """
        Explore phase.

        :param explore_timesteps: Total number of actions used for exploration
        """
        print("explore")
        callbacks = []
        self.explore_model.learn(explore_timesteps, callback=callbacks, use_random_action=True, reset_num_timesteps=False)

    def set_task(self, compute_success: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Robustify, considering the given task

        :param compute_success: Function used to determine wether an observation is a success under the given task
        """
        self.compute_success = compute_success
        self._goal_trajectory = None

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the model's action(s) from an observation

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if self.compute_success is None:
            raise AttributeError("You need to set a task with go_explore.set_task(compute_success)")
        if self._goal_trajectory is None:
            self._goal_trajectory = self.encountered_buffer.plan_trajectory(observation, self.compute_success)
            self._subgoal_idx = 0
        current_cell = self.cell_computer.compute_cell(observation)
        desired_goal = self._goal_trajectory[self._subgoal_idx]
        goal_cell = self.cell_computer.compute_cell(self._goal_trajectory[self._subgoal_idx])
        if current_cell == goal_cell:
            self._subgoal_idx += 1
            desired_goal = self._goal_trajectory[self._subgoal_idx]
        achieved_goal = observation.copy()
        action, _ = self.model.predict(
            {"observation": observation, "desired_goal": desired_goal, "achieved_goal": achieved_goal},
            state,
            mask,
            deterministic,
        )
        return action[0]  # need  [0] since env is vectorizes

    def learn_task(self, model_cls: Type[OffPolicyAlgorithm], task, gradient_steps: int, batch_size: int = 64):
        model = model_cls("MlpPolicy", self.inner_env, verbose=1)
        replay_buffer = self.encountered_buffer.copy()
        replay_buffer.rewards = task(replay_buffer.next_observations)
        model.replay_buffer = replay_buffer
        model._setup_learn(gradient_steps, self.inner_env)
        model.train(gradient_steps, batch_size)
        return model
