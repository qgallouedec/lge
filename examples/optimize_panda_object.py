"""Optimize hyperparameters for PandaPickAndPlace-v1

# Usage :
Can be run in parallel on many workers:
$ python examples/optimize_panda_object.py >> out.log 2>&1 &
"""

import gym
import numpy as np
import optuna
import panda_gym
from go_explore.cell_computers import PandaObjectCellComputer
from go_explore.go_explore import GoExplore
from go_explore.wrapper import UnGoalWrapper
from stable_baselines3 import HerReplayBuffer


def objective(trial: optuna.Study):
    env = gym.make("PandaPickAndPlace-v1")
    env = UnGoalWrapper(env)
    cell_computer = PandaObjectCellComputer()

    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024])
    tau = trial.suggest_categorical("tau", [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    train_freq = trial.suggest_categorical("train_freq", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    net_arch = tuple([[nb_neur for _ in range(depth)] for nb_neur in [32, 64, 128, 256] for depth in [2, 3]])
    net_arch = trial.suggest_categorical("net_arch", net_arch)
    n_sampled_goal = trial.suggest_categorical("n_sampled_goal", [1, 2, 3, 4, 5, 6])
    goal_selection_strategy = trial.suggest_categorical("goal_selection_strategy", ["future", "episode"])
    online_sampling = trial.suggest_categorical("online_sampling", [True, False])
    horizon = trial.suggest_categorical("horizon", [1, 2, 3, 5, 7, 10])
    count_pow = trial.suggest_categorical("count_pow", [0, 1, 2, 3, 4])

    nb_cells = []
    for _ in range(6):
        go_explore = GoExplore(
            env=env,
            cell_computer=cell_computer,
            explore_timesteps=0,
            horizon=horizon,
            count_pow=count_pow,
            learning_rate=learning_rate,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            policy_kwargs=dict(net_arch=net_arch),
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={
                "n_sampled_goal": n_sampled_goal,
                "goal_selection_strategy": goal_selection_strategy,
                "online_sampling": online_sampling,
                "max_episode_length": 50,
            },
        )
        go_explore.exploration(20000)

        nb_cells.append(go_explore.encountered_buffer.nb_cells)
    return np.median(nb_cells)


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///optimize_panda_object.db", study_name="ExploreObject", direction="maximize", load_if_exists=True
    )
    study.optimize(objective, n_trials=5)
