import gym
import numpy as np
import optuna
from go_explore.go_explore.archive import ArchiveBuffer
from go_explore.go_explore.cell_computers import CellIsObs
from go_explore.icm import ICM
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def objective(trial: optuna.Study):
    beta = trial.suggest_categorical("beta", [0, 0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 0.999, 1])
    scaling_factor = trial.suggest_loguniform("scaling_factor", 1e-3, 1e3)
    lmbda = trial.suggest_loguniform("lmbda", 1e-3, 1e3)
    feature_dim = trial.suggest_categorical("feature_dim", [8, 16, 32, 64, 128, 256])
    hidden_dim = trial.suggest_categorical("hidden_dim", [8, 16, 32, 64, 128, 256])

    results = []
    for _ in range(3):
        env = DummyVecEnv([lambda: gym.make("ContinuousMinigrid-v0")])
        env = VecNormalize(env, norm_reward=False)
        icm = ICM(
            beta=beta,
            scaling_factor=scaling_factor,
            lmbda=lmbda,
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
        )
        model = SAC("MlpPolicy", env, reward_modifier=icm, actor_loss_modifier=icm)
        model.replay_buffer = ArchiveBuffer(1000000, env.observation_space, env.action_space, CellIsObs())
        model.learn(10000)
        results.append(model.replay_buffer.nb_cells)

    return np.median(results)


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///example.db", direction="maximize", study_name="PandaReachICMexplore", load_if_exists=True
    )
    study.optimize(objective, n_trials=10)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
