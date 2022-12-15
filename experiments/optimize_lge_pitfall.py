# pip install ale-py==0.7.4
import numpy as np
import optuna
from stable_baselines3 import DDPG, DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from toolbox.maze_grid import compute_coverage
from wandb.integration.sb3 import WandbCallback

import wandb
from experiments.utils import MaxRewardLogger, RAMtoInfoWrapper
from lge import LatentGoExplore

NUM_TIMESTEPS = 100_000
NUM_RUN = 5


env_id = "Pitfall-v4"
module_type = "ae"
n_envs = 2


class NumberRoomsLogger(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.unique_rooms = set()

    def _on_step(self) -> bool:
        buffer = self.locals["replay_buffer"]  # type: ReplayBuffer
        infos = buffer.infos
        if not buffer.full:
            infos = buffer.infos[: buffer.pos]
        rooms = [info[env_idx]["ram"][1] for info in infos for env_idx in range(buffer.n_envs)]
        unique_rooms = set(rooms)
        self.unique_rooms = self.unique_rooms.union(unique_rooms)
        self.logger.record("env/explored rooms", len(self.unique_rooms))
        return True


def objective(trial: optuna.Trial) -> float:
    latent_size = trial.suggest_categorical("latent_size", [4, 8, 16, 32])
    distance_threshold = trial.suggest_categorical("distance_threshold", [0.1, 0.2, 0.5, 1.0])
    lighten_dist_coef = trial.suggest_categorical("lighten_dist_coef", [1.0, 2.0, 4.0, 8.0])
    p = trial.suggest_categorical("p", [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    module_type = trial.suggest_categorical("module_type", ["ae", "inverse", "forward"])

    coverage = np.zeros(NUM_RUN)
    for run_idx in range(NUM_RUN):
        run = wandb.init(
            name=f"lge__{env_id}__{module_type}",
            project="lge",
            config=dict(
                env_id=env_id,
                module_type=module_type,
                latent_size=latent_size,
                distance_threshold=distance_threshold,
                lighten_dist_coef=lighten_dist_coef,
                p=p,
                n_envs=n_envs,
            ),
            sync_tensorboard=True,
        )
        model = LatentGoExplore(
            DQN,
            env_id,
            module_type,
            latent_size,
            distance_threshold,
            lighten_dist_coef,
            p,
            n_envs,
            model_kwargs=dict(buffer_size=100_000, policy_kwargs=dict(categorical=True)),
            wrapper_cls=RAMtoInfoWrapper,
            tensorboard_log=f"runs/{run.id}",
            verbose=1,
        )
        wandb_callback = WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        )
        room_logger = NumberRoomsLogger()
        model.explore(NUM_TIMESTEPS, CallbackList([room_logger, MaxRewardLogger(), wandb_callback]))
        run.finish()
        coverage[run_idx] = len(room_logger.unique_rooms)

    score = np.median(coverage)
    return score


if __name__ == "__main__":
    from optuna.samplers import TPESampler

    study = optuna.create_study(
        storage="sqlite:///optuna.db",
        study_name="lge_pitfall",
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(n_startup_trials=25),
    )
    study.optimize(objective, n_trials=50)
    print(study.best_params, study.best_value)
