from typing import Optional, Tuple, Type

import gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from go_explore.common.callbacks import LogNbCellsCallback
from go_explore.envs import SubgoalEnv
from go_explore.go_explore.cell_computers import CellComputer


class GoExplore:
    """
    Go-Explore paradigma, with DDPG.

    :param env: The environment to learn from (if registered in Gym, can be str)
    :param cell_computer: [description]
    :param subgoal_horizon: [description], defaults to 1
    :param done_delay: [description], defaults to 0
    :param count_pow: [description], defaults to 0
    :param verbose: [description], defaults to 0
    """

    def __init__(
        self,
        env: gym.Env,
        cell_computer: CellComputer,
        subgoal_horizon: int = 1,
        done_delay: int = 0,
        count_pow: int = 0,
        gradient_steps: int = -1,
        verbose: int = 0,
    ) -> None:
        env = SubgoalEnv(env, cell_computer, subgoal_horizon, done_delay, count_pow)
        self.archive = env.archive
        env = DummyVecEnv([lambda: env])
        self.env = VecNormalize(env, norm_reward=False)
        self.model = DDPG(
            "MultiInputPolicy",
            self.env,
            replay_buffer_class=HerReplayBuffer,
            action_noise=OrnsteinUhlenbeckActionNoise(np.zeros(env.action_space.shape[0]), np.ones(env.action_space.shape[0])),
            gradient_steps=gradient_steps,
            verbose=verbose,
        )
        self.model.replay_buffer.child_buffer = self.archive

    def exploration(
        self, total_timesteps: int, eval_freq: int = -1, n_eval_episodes: int = 5, callback: MaybeCallback = None
    ) -> None:
        """
        Exploration phase, switch between go and explore.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param eval_freq: Evaluate the agent every eval_freq timesteps (this may vary a little)
        :param n_eval_episodes: Number of episode to evaluate the agent
        """
        nb_cells_logger = LogNbCellsCallback(self.archive)
        if isinstance(callback, list):
            callback = CallbackList(callback + [nb_cells_logger])
        elif isinstance(callback, BaseCallback):
            callback = CallbackList([callback, nb_cells_logger])
        elif callback is None:
            callback = CallbackList([nb_cells_logger])
        self.model.learn(
            total_timesteps, eval_env=self.env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, callback=callback
        )

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
