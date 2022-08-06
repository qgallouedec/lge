import gym
import gym_continuous_maze
import numpy as np
import optuna
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from toolbox.maze_grid import compute_coverage

from lge import LatentGoExplore

NUM_TIMESTEPS = 50_000
NUM_RUN = 5


def objective(trial: optuna.Trial) -> float:
    distance_threshold = trial.suggest_categorical("distance_threshold", [0.1, 0.2, 0.5, 1.0])
    p = trial.suggest_categorical("p", [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    latent_size = trial.suggest_categorical("latent_size", [2, 4, 8, 16])
    lighten_dist_coef = trial.suggest_categorical("lighten_dist_coef", [1.0, 2.0, 4.0, 8.0])

    coverage = np.zeros((NUM_RUN, NUM_TIMESTEPS))
    for run_idx in range(NUM_RUN):
        env = gym.make("ContinuousMaze-v0")
        model = LatentGoExplore(
            DDPG,
            env,
            distance_threshold=distance_threshold,
            p=p,
            lighten_dist_coef=lighten_dist_coef,
            module_type="inverse",
            latent_size=latent_size,
            model_kwargs=dict(
                buffer_size=NUM_TIMESTEPS,
                action_noise=OrnsteinUhlenbeckActionNoise(
                    np.zeros(env.action_space.shape[0]), np.ones(env.action_space.shape[0])
                ),
            ),
            verbose=1,
        )
        model.explore(NUM_TIMESTEPS)
        buffer = model.archive
        observations = buffer.next_observations["observation"][: buffer.pos if not buffer.full else buffer.buffer_size]
        coverage[run_idx] = compute_coverage(observations) / (24 * 24) * 100

    score = np.median(coverage[:, -1])
    return score


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///optuna.db", study_name="lge_maze", load_if_exists=True, direction="maximize"
    )
    study.optimize(objective, n_trials=30)
    print(study.best_params, study.best_value)
