import go_explore.envs
import gym
import numpy as np
import optuna
from go_explore.go_explore.archive import ArchiveBuffer
from go_explore.go_explore.cell_computers import CellIsObs
from go_explore.rnd import RND, PredictorLearner
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def objective(trial: optuna.Study):
    scaling_factor = trial.suggest_categorical("scaling_factor", [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
    train_freq = trial.suggest_categorical("train_freq", [1, 5, 10, 50, 100, 500, 1000])
    grad_step = trial.suggest_categorical("grad_step", [1, 5, 10, 50, 100, 500, 1000])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    lr = trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    out_dim = trial.suggest_categorical("out_dim", [16, 32, 64, 128, 256, 512])

    if grad_step / train_freq > 1:
        return 0  # otherwise, learning is very slow

    results = []
    for _ in range(3):
        env = DummyVecEnv(8 * [lambda: gym.make("ContinuousMinigrid-v0")])
        env = VecNormalize(env, norm_reward=False)
        rnd = RND(
            scaling_factor=scaling_factor, obs_dim=env.observation_space.shape[0], out_dim=out_dim, hidden_dim=hidden_dim
        )
        model = SAC("MlpPolicy", env, replay_buffer_class=ArchiveBuffer, replay_buffer_kwargs={"cell_computer": CellIsObs()})
        learner = PredictorLearner(
            predictor=rnd.predictor,
            target=rnd.target,
            buffer=model.replay_buffer,
            train_freq=train_freq,
            grad_step=grad_step,
            weight_decay=weight_decay,
            lr=lr,
            batch_size=batch_size,
        )
        model.learn(10000, callback=learner, reward_modifier=rnd)
        results.append(model.replay_buffer.nb_cells)

    return np.median(results)


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///example.db", direction="maximize", study_name="PandaReachRNDexplore", load_if_exists=True
    )
    study.optimize(objective, n_trials=10)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
