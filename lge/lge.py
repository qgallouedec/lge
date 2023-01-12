import copy
from typing import Any, Dict, Optional, Type, Union

import gym
import numpy as np
import torch
from gym import spaces
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecEnvWrapper, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from lge.buffer import LGEBuffer
from lge.learners import AEModuleLearner, ForwardModuleLearner, InverseModuleLearner, VQVAEModuleLearner, VQVAEForwardModuleLearner
from lge.modules.ae_module import AEModule, VQVAEModule
from lge.modules.forward_module import VQVAEForwardModule, ForwardModule
from lge.modules.inverse_module import CNNInverseModule, InverseModule
from lge.utils import get_shape, get_size, maybe_make_channel_first, maybe_transpose


class VecGoalify(VecEnvWrapper):
    """
    Wrap the env into a GoalEnv.

    :param venv: The vectorized environment
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
        venv: VecEnv,
        nb_random_exploration_steps: int = 30,
        window_size: int = 10,
        distance_threshold: float = 1.0,
        lighten_dist_coef: float = 1.0,
    ) -> None:
        super().__init__(venv)
        self.observation_space = spaces.Dict(
            {
                "observation": copy.deepcopy(venv.observation_space),
                "goal": copy.deepcopy(venv.observation_space),
            }
        )
        self.nb_random_exploration_steps = nb_random_exploration_steps
        self.window_size = window_size
        self.distance_threshold = distance_threshold
        self.lighten_dist_coef = lighten_dist_coef
        self.lge_buffer: LGEBuffer
        self.goal_trajectories = [None for _ in range(self.num_envs)]
        self.emb_trajectories = [None for _ in range(self.num_envs)]

    def reset(self) -> VecEnvObs:
        observations = self.venv.reset()
        assert hasattr(self, "lge_buffer"), "you need to set the buffer before reset. Use set_buffer()"
        for env_idx in range(self.num_envs):
            goal_trajectory, emb_trajectory = self.lge_buffer.sample_trajectory(self.lighten_dist_coef)
            # For image, we need to transpose the sample
            goal_trajectory = maybe_transpose(goal_trajectory, self.observation_space["goal"])
            self.goal_trajectories[env_idx] = goal_trajectory
            self.emb_trajectories[env_idx] = emb_trajectory

        self._goal_idxs = np.zeros(self.num_envs, dtype=np.int64)
        self.done_countdowns = self.nb_random_exploration_steps * np.ones(self.num_envs, dtype=np.int64)
        self._is_last_goal_reached = np.zeros(self.num_envs, dtype=bool)  # useful flag
        dict_observations = self._get_dict_obs(observations)  # turn into dict
        return dict_observations

    def set_buffer(self, lge_buffer: LGEBuffer) -> None:
        """
        Set the buffer.

        The buffer is used to compute goal trajectories, and the latent representation.

        :param buffer: The LGE buffer
        """
        self.lge_buffer = lge_buffer

    def _get_dict_obs(self, observations: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "observation": observations,
            "goal": np.stack([self.goal_trajectories[env_idx][self._goal_idxs[env_idx]] for env_idx in range(self.num_envs)]),
        }

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions
        return super().step_async(actions)

    def _reset_one_env(self, env_idx: int):
        if isinstance(self.venv, SubprocVecEnv):
            self.venv.remotes[env_idx].send(("reset", None))
            return self.venv.remotes[env_idx].recv()
        if isinstance(self.venv, DummyVecEnv):
            return self.venv.envs[env_idx].reset()

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()

        for info, reward in zip(infos, rewards):
            info["env_reward"] = reward

        # Move to next goal here (by modifying self._goal_idx and self._is_last_goal_reached)
        embeddings = self.lge_buffer.encode(maybe_make_channel_first(observations))
        for env_idx in range(self.num_envs):
            infos[env_idx]["is_success"] = self._is_last_goal_reached[env_idx]  # Will be overwritten if necessary
            if not dones[env_idx]:
                upper_idx = min(self._goal_idxs[env_idx] + self.window_size, len(self.goal_trajectories[env_idx]))
                future_goals = self.emb_trajectories[env_idx][self._goal_idxs[env_idx] : upper_idx]
                dist = np.linalg.norm(embeddings[env_idx] - future_goals, axis=1)
                future_success = dist < self.distance_threshold

                if future_success.any():
                    furthest_futur_success = np.where(future_success)[0].max()
                    self._goal_idxs[env_idx] += furthest_futur_success + 1
                if self._goal_idxs[env_idx] == len(self.goal_trajectories[env_idx]):
                    self._is_last_goal_reached[env_idx] = True
                    self._goal_idxs[env_idx] -= 1

                # When the last goal is reached, delay the done to allow some random actions
                if self._is_last_goal_reached[env_idx]:
                    infos[env_idx]["is_success"] = True
                    if self.done_countdowns[env_idx] != 0:
                        infos[env_idx]["action_repeat"] = self.actions[env_idx]
                        self.done_countdowns[env_idx] -= 1
                    else:  # self.done_countdown == 0:
                        dones[env_idx] = True
                        terminal_observation = observations[env_idx]
                        observations[env_idx] = self._reset_one_env(env_idx)

            # Dones can be due to env (death), or to the previous code
            if dones[env_idx]:
                # If done is due to inner env, terminal obs is already in infos. Else
                # it is written in terminal obs, see above.
                if "terminal_observation" in infos[env_idx]:
                    terminal_observation = infos[env_idx]["terminal_observation"]
                infos[env_idx]["terminal_observation"] = {
                    "observation": terminal_observation,
                    "goal": self.goal_trajectories[env_idx][self._goal_idxs[env_idx]],
                }
                goal_trajectory, emb_trajectory = self.lge_buffer.sample_trajectory(self.lighten_dist_coef)
                # For image, we need to transpose the sample
                goal_trajectory = maybe_transpose(goal_trajectory, self.observation_space["goal"])
                self.goal_trajectories[env_idx] = goal_trajectory
                self.emb_trajectories[env_idx] = emb_trajectory
                self._goal_idxs[env_idx] = 0
                self.done_countdowns[env_idx] = self.nb_random_exploration_steps
                self._is_last_goal_reached[env_idx] = False

        dict_observations = self._get_dict_obs(observations)
        return dict_observations, rewards, dones, infos


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
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation, defaults to None
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param model_kwargs: Keyword arguments to pass to the model on creation, defaults to None
    :param wrapper_cls: Wrapper class.
    :param nb_random_exploration_steps: Number of random exploration steps
    :param module_train_freq: Module train frequency
    :param module_grad_steps: Module gradient steps per training rollout
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
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
        env_kwargs: Optional[Dict[str, Any]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        learning_starts: int = 100,
        model_kwargs: Optional[Dict[str, Any]] = None,
        wrapper_cls: Optional[gym.Wrapper] = None,
        nb_random_exploration_steps: int = 50,
        module_train_freq: int = 5_000,
        module_grad_steps: int = 500,
        vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[torch.device, str] = "auto",
    ) -> None:
        self.device = get_device(device)

        env_kwargs = {} if env_kwargs is None else env_kwargs
        venv = make_vec_env(env_id, n_envs=n_envs, wrapper_class=wrapper_cls, env_kwargs=env_kwargs, vec_env_cls=vec_env_cls)
        venv = VecGoalify(
            venv,
            distance_threshold=distance_threshold,
            nb_random_exploration_steps=nb_random_exploration_steps,
            lighten_dist_coef=lighten_dist_coef,
        )
        venv = VecMonitor(venv)

        # Define the "module" used to learn the latent representation
        action_size = get_size(venv.action_space)
        if is_image_space(venv.observation_space["observation"]):
            obs_shape = get_shape(venv.observation_space["observation"])
            if module_type == "inverse":
                self.module = CNNInverseModule(obs_shape, action_size, latent_size).to(self.device)
            elif module_type == "forward":
                self.module = VQVAEForwardModule(action_size).to(self.device)
                latent_size = self.module.vqvae.vq_layer.num_embeddings * 8 * 8
            elif module_type == "ae":
                self.module = VQVAEModule().to(self.device)
                # Rewriting on latent size
                latent_size = self.module.vqvae.vq_layer.num_embeddings * 8 * 8
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
        model_kwargs["learning_starts"] = learning_starts
        model_kwargs["train_freq"] = 10
        model_kwargs["gradient_steps"] = n_envs
        model_kwargs["tensorboard_log"] = tensorboard_log
        self.model = model_class(
            "MultiInputPolicy",
            venv,
            replay_buffer_class=LGEBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            verbose=verbose,
            **model_kwargs,
        )
        self.replay_buffer = self.model.replay_buffer  # type: LGEBuffer

        venv.set_buffer(self.replay_buffer)

        # Define the learner for module
        if module_type == "inverse":
            self.module_learner = InverseModuleLearner(
                self.module,
                self.replay_buffer,
                train_freq=module_train_freq,
                gradient_steps=module_grad_steps,
                learning_starts=learning_starts,
            )
        elif module_type == "forward":
            if is_image_space(venv.observation_space["observation"]):
                self.module_learner = VQVAEForwardModuleLearner(
                    self.module,
                    self.replay_buffer,
                    train_freq=module_train_freq,
                    gradient_steps=module_grad_steps,
                    learning_starts=learning_starts,
                )
            else:
                self.module_learner = ForwardModuleLearner(
                    self.module,
                    self.replay_buffer,
                    train_freq=module_train_freq,
                    gradient_steps=module_grad_steps,
                    learning_starts=learning_starts,
                )
        elif module_type == "ae":
            if is_image_space(venv.observation_space["observation"]):
                self.module_learner = VQVAEModuleLearner(
                    self.module,
                    self.replay_buffer,
                    train_freq=module_train_freq,
                    gradient_steps=module_grad_steps,
                    learning_starts=learning_starts,
                )
            else:
                self.module_learner = AEModuleLearner(
                    self.module,
                    self.replay_buffer,
                    train_freq=module_train_freq,
                    gradient_steps=module_grad_steps,
                    learning_starts=learning_starts,
                )

    def explore(self, total_timesteps: int, callback: MaybeCallback = None) -> None:
        """
        Run exploration.

        :param total_timesteps: Total number of timesteps for exploration
        """
        if callback is not None:
            callback = [self.module_learner, callback]
        else:
            callback = [self.module_learner]
        self.model.learn(total_timesteps, callback=callback, log_interval=1000)
