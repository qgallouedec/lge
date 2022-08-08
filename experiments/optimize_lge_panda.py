import gym
import numpy as np
import optuna
import panda_gym
from stable_baselines3 import DDPG
from toolbox.panda_utils import cumulative_object_coverage

from lge import LatentGoExplore

NUM_TIMESTEPS = 300_000
NUM_RUN = 3


def objective(trial: optuna.Trial) -> float:
    distance_threshold = trial.suggest_categorical("distance_threshold", [0.1, 0.2, 0.5, 1.0])
    p = trial.suggest_categorical("p", [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    latent_size = trial.suggest_categorical("latent_size", [8, 16, 32, 64])
    lighten_dist_coef = trial.suggest_categorical("lighten_dist_coef", [1, 2, 4, 8])

    coverage = np.zeros((NUM_RUN, NUM_TIMESTEPS))
    for run_idx in range(NUM_RUN):
        env = gym.make("PandaNoTask-v0", nb_objects=1)
        model = LatentGoExplore(
            DDPG,
            env,
            distance_threshold=distance_threshold,
            p=p,
            lighten_dist_coef=lighten_dist_coef,
            module_type="forward",
            latent_size=latent_size,
            model_kwargs=dict(buffer_size=NUM_TIMESTEPS),
            verbose=1,
        )
        model.explore(NUM_TIMESTEPS)
        buffer = model.archive
        observations = buffer.next_observations["observation"][: buffer.pos if not buffer.full else buffer.buffer_size]
        coverage[run_idx] = cumulative_object_coverage(observations)

    score = np.median(coverage[:, -1])
    return score


if __name__ == "__main__":
    from optuna.samplers import TPESampler

    study = optuna.create_study(
        storage="sqlite:///optuna.db",
        study_name="lge_panda",
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(n_startup_trials=25),
    )
    study.optimize(objective, n_trials=50)
    print(study.best_params, study.best_value)
