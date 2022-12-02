import copy
from typing import Any, Dict, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
from gym import Env, spaces
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.utils import get_device

from lge.buffer import LGEBuffer
from lge.learners import AEModuleLearner, ForwardModuleLearner, InverseModuleLearner
from lge.modules.ae_module import AEModule, CNNAEModule
from lge.modules.forward_module import CNNForwardModule, ForwardModule
from lge.modules.inverse_module import CNNInverseModule, InverseModule
from lge.utils import get_shape, get_size


class Goalify(gym.Wrapper):
    """
    Wrap the env into a GoalEnv.

    :param env: The environment
    :param nb_random_exploration_steps: Number of random exploration steps after the goal is reached, defaults to 30
    :param window_size: Agent can skip goals in the goal trajectory within the limit of ``window_size``
        goals ahead, defaults to 10
    :param distance_threshold: The goal is reached when the latent distance between
    the current obs and the goal obs is below this threshold, defaults to 1.0
    :param lighten_dist_coef: Remove subgoal that are not further than lighten_dist_coef*dist_threshold
        from the previous subgoal, defaults to 1.0
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
        self.lge_buffer: LGEBuffer
        self.nb_random_exploration_steps = nb_random_exploration_steps
        self.window_size = window_size
        self.distance_threshold = distance_threshold
        self.lighten_dist_coef = lighten_dist_coef

    def set_buffer(self, lge_buffer: LGEBuffer) -> None:
        """
        Set the buffer.

        The buffer is used to compute goal trajectories, and the latent representation.

        :param buffer: The LGE buffer
        """
        self.lge_buffer = lge_buffer

    def reset(self) -> Dict[str, np.ndarray]:
        obs = self.env.reset()
        assert hasattr(self, "lge_buffer"), "you need to set the buffer before reset. Use set_buffer()"
        self.goal_trajectory, self.emb_trajectory = self.lge_buffer.sample_trajectory(self.lighten_dist_coef)
        self._goal_idx = 0
        self.done_countdown = self.nb_random_exploration_steps
        self._is_last_goal_reached = False  # useful flag
        dict_obs = self._get_dict_obs(obs)  # turn into dict
        return dict_obs

    def _get_dict_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "observation": obs,
            "goal": self.goal_trajectory[self._goal_idx],
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
    """
    Latent Go-Explore.

    :param model_class: Off-policy algorithm of the goal-conditioned agent
     :param env: The environment to learn from
    :param module_type: Type of module used for learning the representation
        in ["inverse", "forward", "ae"], defaults to "inverse"
    :param latent_size: Feature size, defaults to 16
    :param distance_threshold: The goal is reached when the latent distance between
        the current obs and the goal obs is below this threshold, defaults to 1.0
    :param lighten_dist_coef: Remove subgoal that are not further than lighten_dist_coef*dist_threshold
        from the previous subgoal, defaults to 1.0
    :param p: Geometric parameter for final goal sampling, defaults to 0.005
    :param n_envs: Number of parallel environments
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation, defaults to None
    :param model_kwargs: Keyword arguments to pass to the model on creation, defaults to None
    :param further_explore: Whether the agent further explore after reaching the final goal, defaults to True
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug, defaults to 0
    :param device: PyTorch device, defaults to "auto"
    """

    def __init__(
        self,
        model_class: Type[OffPolicyAlgorithm],
        env_id: str,
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
        self.device = get_device(device)

        # Wrap the env
        def env_func():
            return Goalify(
                gym.make(env_id, verbose),
                distance_threshold=distance_threshold,
                nb_random_exploration_steps=50 if further_explore else 0,
                lighten_dist_coef=lighten_dist_coef,
            )

        venv = make_vec_env(env_func, n_envs=n_envs)

        # Define the "module" used to learn the latent representation
        action_size = get_size(venv.action_space)
        if is_image_space(venv.observation_space["observation"]):
            obs_shape = get_shape(venv.observation_space["observation"])
            if module_type == "inverse":
                self.module = CNNInverseModule(obs_shape, action_size, latent_size).to(self.device)
            elif module_type == "forward":
                self.module = CNNForwardModule(obs_shape, action_size, latent_size).to(self.device)
            elif module_type == "ae":
                self.module = CNNAEModule(obs_shape, latent_size).to(self.device)
        else:  # Not image
            obs_size = get_size(venv.observation_space["observation"])
            if module_type == "inverse":
                self.module = InverseModule(obs_size, action_size, latent_size).to(self.device)
            elif module_type == "forward":
                self.module = ForwardModule(obs_size, action_size, latent_size).to(self.device)
            elif module_type == "ae":
                self.module = AEModule(obs_size, latent_size).to(self.device)

        replay_buffer_kwargs = {} if replay_buffer_kwargs is None else replay_buffer_kwargs
        replay_buffer_kwargs.update(
            dict(encoder=self.module.encoder, latent_size=latent_size, distance_threshold=distance_threshold, p=p)
        )
        model_kwargs = {} if model_kwargs is None else model_kwargs
        model_kwargs["learning_starts"] = 3_000
        model_kwargs["train_freq"] = 1
        model_kwargs["gradient_steps"] = 1
        self.model = model_class(
            "MultiInputPolicy",
            venv,
            replay_buffer_class=LGEBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            verbose=verbose,
            **model_kwargs,
        )
        self.replay_buffer = self.model.replay_buffer  # type: LGEBuffer
        for _env in self.model.env.envs:
            _env.set_buffer(self.replay_buffer)

        # Define the learner for module
        if module_type == "inverse":
            self.module_learner = InverseModuleLearner(self.module, self.replay_buffer)
        elif module_type == "forward":
            self.module_learner = ForwardModuleLearner(self.module, self.replay_buffer)
        elif module_type == "ae":
            self.module_learner = AEModuleLearner(self.module, self.replay_buffer)

    def explore(self, total_timesteps: int) -> None:
        """
        Run exploration.

        :param total_timesteps: Total number of timesteps for exploration
        """
        self.model.learn(total_timesteps, callback=[self.module_learner])
