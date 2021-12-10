import go_explore.envs
import gym
import numpy as np
import optuna
from go_explore.go_explore.archive import ArchiveBuffer
from go_explore.go_explore.cell_computers import CellIsObs
from go_explore.simhash import SimHashMotivation
from go_explore.simhash.simhash import SimHashMotivation
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def objective(trial: optuna.Study):
    granularity = trial.suggest_categorical("granularity", [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    beta = trial.suggest_loguniform("beta", 1e-3, 1e3)

    results = []
    for _ in range(3):
        env = DummyVecEnv(8 * [lambda: gym.make("ContinuousMinigrid-v0")])
        env = VecNormalize(env, norm_reward=False)
        model = SAC("MlpPolicy", env, replay_buffer_class=ArchiveBuffer, replay_buffer_kwargs={"cell_computer": CellIsObs()})
        simhash = SimHashMotivation(model.replay_buffer, env, granularity, beta)
        model.learn(10000, reward_modifier=simhash)
        results.append(model.replay_buffer.nb_cells)

    return np.median(results)


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///example.db", direction="maximize", study_name="PandaReachSimHashexplore", load_if_exists=True
    )
    study.optimize(objective, n_trials=10)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
