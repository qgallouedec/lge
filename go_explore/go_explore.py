from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3 import DDPG
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.td3.policies import TD3Policy

from go_explore.buffer import PathfinderBuffer
from go_explore.callback import LogNbCellsCallback, SaveNbCellsCallback, StopTrainingOnEndTrajectory, StoreCallback
from go_explore.cell_computers import CellComputer
from go_explore.wrapper import GoalBufferTrajcetorySetterWrapper, GoExploreWrapper, HardResetSometimesWrapper


class GoExplore:
    """
    Go-Explore paradigma, with DDPG.

    :param env: The environment to learn from (if registered in Gym, can be str)
    :param cell_computer: The cell computer
    :param go_timesteps: The max number of timesteps before forcing to stop the explore phase (and move to the explore phase)
    :param explore_timesteps: The number of random move when in explore phase
    :param horizon: When a subgoal trajectory is generated, you can choose to ignore some some subgoal. Set to ``1`` means
        no subgoal is ignored. Set to ``2``means one subgoal up to two is ignored
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        cell_computer: CellComputer,
        go_timesteps: int,
        explore_timesteps: int,
        horizon: int = 1,
        count_pow: int = 1,
        policy: Union[str, Type[TD3Policy]] = "MultiInputPolicy",
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        self.inner_env = env
        self.cell_computer = cell_computer
        env = GoExploreWrapper(env, cell_computer)
        env = TimeLimit(env, max_episode_steps=50)

        self.go_timesteps = go_timesteps
        self.explore_timesteps = explore_timesteps

        self.goal_buffer = PathfinderBuffer(
            buffer_size, env.observation_space, env.action_space, self.cell_computer, horizon, count_pow
        )
        self.encountered_buffer = self.goal_buffer
        env = GoalBufferTrajcetorySetterWrapper(env, self.goal_buffer)
        env.hard_reset()
        env = HardResetSometimesWrapper(env, 300)
        self.env = env

        self.save_nb_cells_callback = SaveNbCellsCallback(self.encountered_buffer, 500)

        self.compute_success = None
        self._goal_trajectory = None  # used for predict

        self.model = DDPG(
            policy=policy,
            env=self.env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            _init_setup_model=_init_setup_model,
        )

    def exploration(self, total_timesteps: int) -> None:
        """
        Exploration phase, switch between go and explore.

        :param total_timesteps: Total number of timesteps
        """
        while self.model.num_timesteps < total_timesteps:
            try:
                self.go(total_timesteps)
            except IndexError:
                pass
            self.explore(50)

    def go(self, total_timesteps: int) -> None:
        """
        Go phase. Ends when a trajectory is entirely acheived.

        :param total_timesteps: Max number of timesteps. Learning stops early if a final goal is reached
        """
        callbacks = [
            StoreCallback(self.encountered_buffer),
            LogNbCellsCallback(self.encountered_buffer),
            StopTrainingOnEndTrajectory(),
            self.save_nb_cells_callback,
        ]
        self.model.learn(
            total_timesteps,
            eval_env=self.env,
            eval_freq=2000,
            n_eval_episodes=100,
            reset_num_timesteps=False,
            callback=callbacks,
        )

    def explore(self, explore_timesteps: int) -> None:
        """
        Explore phase.

        :param explore_timesteps: Total number of actions used for exploration
        """
        callbacks = [
            StoreCallback(self.encountered_buffer),
            LogNbCellsCallback(self.encountered_buffer),
            self.save_nb_cells_callback,
        ]
        self.model.learn(explore_timesteps, callback=callbacks, use_random_action=True, reset_num_timesteps=False)

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
