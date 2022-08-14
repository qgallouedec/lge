import copy
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
from gym import Env, spaces
from stable_baselines3.common.base_class import maybe_make_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.utils import get_device

from lge.buffer import LGEBuffer
from lge.learners import AEModuleLearner, ForwardModuleLearner, InverseModuleLearner
from lge.modules.ae_module import AEModule
from lge.modules.forward_module import ForwardModule
from lge.modules.inverse_module import InverseModule


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
        distance_threshold: float = 1.0,
        lighten_dist_coef: float = 1.0,
    ) -> None:
        super().__init__(env)
        # Set a goal-conditionned observation space
        self.observation_space = spaces.Dict(
            {
                "observation": copy.deepcopy(self.env.observation_space),
                "goal": copy.deepcopy(self.env.observation_space),
            }
        )
        self.lge_buffer = None  # type: LGEBuffer
        self.nb_random_exploration_steps = nb_random_exploration_steps
        self.window_size = window_size
        self.distance_threshold = distance_threshold
        self.lighten_dist_coef = lighten_dist_coef

    def set_buffer(self, lge_buffer: LGEBuffer) -> None:
        """
        Set the buffer.

        The buffer is used to compute goal trajectories, and to compute the cell for the reward.

        :param buffer: The LGE buffer
        """
        self.lge_buffer = lge_buffer

    def reset(self) -> Dict[str, np.ndarray]:
        obs = self.env.reset()
        assert self.lge_buffer is not None, "you need to set the buffer before reset. Use set_buffer()"
        self.goal_trajectory, self.emb_trajectory = self.lge_buffer.sample_trajectory(self.lighten_dist_coef)
        if is_image_space(self.observation_space["goal"]):
            self.goal_trajectory = [np.moveaxis(goal, 0, 2) for goal in self.goal_trajectory]
        self._goal_idx = 0
        self.done_countdown = self.nb_random_exploration_steps
        self._is_last_goal_reached = False  # useful flag
        dict_obs = self._get_dict_obs(obs)  # turn into dict
        return dict_obs

    def _get_dict_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "observation": obs.astype(np.float32),
            "goal": self.goal_trajectory[self._goal_idx].astype(np.float32),
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        # Compute reward (has to be done before moving to next goal)
        embedding = self.lge_buffer.encode(obs).detach().cpu().numpy()

        # Move to next goal here (by modifying self._goal_idx and self._is_last_goal_reached)
        upper_idx = min(self._goal_idx + self.window_size, len(self.goal_trajectory))
        future_goals = self.emb_trajectory[self._goal_idx : upper_idx]
        dist = np.linalg.norm(embedding - future_goals, axis=1)
        future_success = dist < self.distance_threshold

        if future_success.any():
            furthest_futur_success = np.where(future_success)[0].max()
            self._goal_idx += furthest_futur_success + 1
        if self._goal_idx == len(self.goal_trajectory):
            self._is_last_goal_reached = True
            self._goal_idx -= 1

        if future_success[0]:
            # Agent has just reached the current goal
            reward = 0
        else:
            # Agent has reached another goal, or no goal at all
            reward = -1

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


class LatentGoExplore:
    """ """

    def __init__(
        self,
        model_class: Type[OffPolicyAlgorithm],
        env: Env,
        module_type: str = "inverse",
        latent_size: int = 16,
        distance_threshold: float = 1.0,
        lighten_dist_coef: float = 1.0,
        p: float = 0.005,
        n_envs: int = 1,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        further_explore: bool = True,
        verbose: int = 0,
        device: Union[torch.device, str] = "auto",
    ) -> None:
        env = maybe_make_env(env, verbose)
        if type(env.action_space) is spaces.Discrete:
            action_size = env.action_space.n
        elif type(env.action_space) is spaces.Box:
            action_size = env.action_space.shape[0]
        obs_size = env.observation_space.shape[0]
        if is_image_space(env.observation_space):
            raise NotImplementedError()

        self.device = get_device(device)

        # Define the "module" used to learn the latent representation
        if module_type == "inverse":
            self.module = InverseModule(obs_size, action_size, latent_size).to(self.device)
            self.module_learner = InverseModuleLearner(self.module, self.replay_buffer)
        elif module_type == "forward":
            self.module = ForwardModule(obs_size, action_size, latent_size).to(self.device)
            self.module_learner = ForwardModuleLearner(self.module, self.replay_buffer)
        elif module_type == "ae":
            self.module = AEModule(obs_size, latent_size).to(self.device)
            self.module_learner = AEModuleLearner(self.module, self.replay_buffer)

        # Wrap the env
        def env_func():
            return Goalify(
                maybe_make_env(env, verbose),
                distance_threshold=distance_threshold,
                nb_random_exploration_steps=50 if further_explore else 0,
                lighten_dist_coef=lighten_dist_coef,
            )

        env = make_vec_env(env_func, n_envs=n_envs)
        replay_buffer_kwargs = {} if replay_buffer_kwargs is None else replay_buffer_kwargs
        replay_buffer_kwargs.update(dict(encoder=self.module.encoder, distance_threshold=distance_threshold, p=p))
        model_kwargs = {} if model_kwargs is None else model_kwargs
        model_kwargs["learning_starts"] = 3_000
        model_kwargs["train_freq"] = 1
        model_kwargs["gradient_steps"] = 1
        self.model = model_class(
            "MultiInputPolicy",
            env,
            replay_buffer_class=LGEBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            verbose=verbose,
            **model_kwargs,
        )
        self.replay_buffer = self.model.replay_buffer  # type: LGEBuffer
        for _env in self.model.env.envs:
            _env.set_buffer(self.replay_buffer)

    def explore(self, total_timesteps: int) -> None:
        """
        Run exploration.

        :param total_timesteps: Total number of timesteps for exploration
        """
        self.model.learn(total_timesteps, callback=[self.module_learner])
