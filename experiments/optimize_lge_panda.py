import numpy as np
import optuna
import panda_gym
from stable_baselines3 import DDPG
from toolbox.panda_utils import compute_coverage

from lge import LatentGoExplore

NUM_TIMESTEPS = 200_000
NUM_RUN = 3


def objective(trial: optuna.Trial) -> float:
    latent_size = trial.suggest_categorical("latent_size", [4, 8, 16, 32])
    distance_threshold = trial.suggest_categorical("distance_threshold", [0.1, 0.2, 0.5, 1.0])
    lighten_dist_coef = trial.suggest_categorical("lighten_dist_coef", [0, 1, 2, 4])
    p = trial.suggest_categorical("p", [0.005, 0.01, 0.02, 0.05, 0.1])

    coverage = np.zeros((NUM_RUN, NUM_TIMESTEPS))
    for run_idx in range(NUM_RUN):
        model = LatentGoExplore(
            DDPG,
            "PandaNoTask-v0",
            module_type="forward",
            latent_size=latent_size,
            distance_threshold=distance_threshold,
            lighten_dist_coef=lighten_dist_coef,
            p=p,
            model_kwargs=dict(buffer_size=NUM_TIMESTEPS),
            verbose=1,
            env_kwargs=dict(nb_objects=1),
        )
        model.explore(NUM_TIMESTEPS)
        buffer = model.replay_buffer
        observations = buffer.next_observations["observation"][: buffer.pos if not buffer.full else buffer.buffer_size]
        coverage[run_idx] = compute_coverage(observations)

    score = np.median(coverage[:, -1])
    return score


if __name__ == "__main__":
    from optuna.samplers import TPESampler

    study = optuna.create_study(
        storage="sqlite:///optuna.db",
        study_name="lge_panda_new_cov",
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(n_startup_trials=25),
    )
    study.optimize(objective, n_trials=50)
    print(study.best_params, study.best_value)
