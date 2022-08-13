import copy
from typing import Any, Callable, Dict, Optional, Tuple, Type

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

from lge.archive import ArchiveBuffer
from lge.feature_extractor import GoExploreExtractor
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
        self.archive = None  # type: ArchiveBuffer
        self.nb_random_exploration_steps = nb_random_exploration_steps
        self.window_size = window_size
        self.distance_threshold = distance_threshold
        self.lighten_dist_coef = lighten_dist_coef

    def set_archive(self, archive: ArchiveBuffer) -> None:
        """
        Set the archive.

        The archive is used to compute goal trajectories, and to compute the cell for the reward.

        :param archive: The archive
        """
        self.archive = archive

    def reset(self) -> Dict[str, np.ndarray]:
        obs = self.env.reset()
        assert self.archive is not None, "you need to set the archive before reset. Use set_archive()"
        self.goal_trajectory, self.emb_trajectory = self.archive.sample_trajectory(self.lighten_dist_coef)
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
        embedding = self.archive.encode(obs).detach().cpu().numpy()

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
    ) -> None:
        env = maybe_make_env(env, verbose)
        if type(env.action_space) is spaces.Discrete:
            action_size = env.action_space.n
        elif type(env.action_space) is spaces.Box:
            action_size = env.action_space.shape[0]
        obs_size = env.observation_space.shape[0]
        if is_image_space(env.observation_space):
            raise NotImplementedError()
        else:
            if module_type == "inverse":
                self.module = InverseModule(obs_size, action_size, latent_size, device=get_device("auto"))
            elif module_type == "forward":
                self.module = ForwardModule(obs_size, action_size, latent_size, device=get_device("auto"))
            elif module_type == "ae":
                self.module = AEModule(obs_size, latent_size, device=get_device("auto"))

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
        policy_kwargs = dict(features_extractor_class=GoExploreExtractor)
        model_kwargs = {} if model_kwargs is None else model_kwargs
        model_kwargs["learning_starts"] = 3_000
        model_kwargs["train_freq"] = 1
        model_kwargs["gradient_steps"] = 1
        self.model = model_class(
            "MultiInputPolicy",
            env,
            replay_buffer_class=ArchiveBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            **model_kwargs,
        )
        self.archive = self.model.replay_buffer  # type: ArchiveBuffer
        for _env in self.model.env.envs:
            _env.set_archive(self.archive)

    def explore(self, total_timesteps: int, train_freq=5_000, gradient_steps=500, reset_num_timesteps: bool = False) -> None:
        """
        Run exploration.

        :param total_timesteps: Total timestep of exploration
        :param update_freq: Cells update frequency
        :param reset_num_timesteps: Whether or not to reset the current timestep number (used in logging), defaults to False
        """
        if type(self.model.env.action_space) == spaces.Discrete:
            criterion = torch.nn.CrossEntropyLoss()
        elif type(self.model.env.action_space) == spaces.Box:
            criterion = torch.nn.MSELoss()

        if isinstance(self.module, InverseModule):
            learner_class = InverseModuleLearner
        elif isinstance(self.module, ForwardModule):
            learner_class = ForwardModuleLearner
        elif isinstance(self.module, AEModule):
            learner_class = AEModuleLearner

        callback = [
            learner_class(
                self.module,
                self.archive,
                criterion=criterion,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                first_update=5_000,
            ),
        ]
        self.model.learn(total_timesteps, callback=callback, reset_num_timesteps=reset_num_timesteps)
