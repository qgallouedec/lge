import gym
import numpy as np
import optuna
from go_explore.go_explore.cell_computers import CellIsObs
from go_explore.go_explore.go_explore import GoExplore


def objective(trial: optuna.Study):
    count_pow = trial.suggest_categorical("count_pow", [0, 1, 2, 3, 4, 5, 6])
    subgoal_horizon = trial.suggest_categorical("subgoal_horizon", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    done_delay = trial.suggest_categorical("done_delay", [0, 1, 2, 4, 8, 16, 32, 64])

    results = []

    for _ in range(5):
        env = gym.make("ContinuousMinigrid-v0")
        ge = GoExplore(env, CellIsObs(), subgoal_horizon, done_delay, count_pow)
        ge.exploration(5000)
        results.append(ge.archive.nb_cells)

    return np.median(results)


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///example.db",
        direction="maximize",
        study_name="PandaReachGoexplore",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=10)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
