import go_explore.envs
import gym
import numpy as np
import optuna
from go_explore.envs import PandaReachFlat
from go_explore.go_explore.archive import ArchiveBuffer
from go_explore.go_explore.cell_computers import CellIsObs
from go_explore.simhash import SimHashMotivation
from go_explore.simhash.simhash import SimHashMotivation
from go_explore.surprise import SurpriseMotivation, TransitionModelLearner
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def objective(trial: optuna.Study):
    eta_0 = trial.suggest_categorical("eta_0", [0.1, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0])
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256, 512])
    train_freq = trial.suggest_categorical("train_freq", [1, 5, 10, 50, 100, 500, 1000])
    grad_step = trial.suggest_categorical("grad_step", [1, 5, 10, 50, 100, 500, 1000])
    # weight_decay = trial.suggest_loguniform("weight_decay", 1e-7, 1e-2)
    lr = trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])

    if grad_step / train_freq > 1:
        return 0  # otherwise, learning is very slow

    results = []
    for _ in range(5):
        env = DummyVecEnv(8 * [lambda: gym.make("ContinuousMinigrid-v0")])
        env = VecNormalize(env, norm_reward=False)
        surprise_motivation = SurpriseMotivation(
            obs_size=env.observation_space.shape[0],
            action_size=env.action_space.shape[0],
            eta_0=eta_0,
            hidden_size=hidden_size,
        )
        model = SAC("MlpPolicy", env, replay_buffer_class=ArchiveBuffer, replay_buffer_kwargs={"cell_computer": CellIsObs()})
        learner = TransitionModelLearner(
            transition_model=surprise_motivation.transition_model,
            buffer=model.replay_buffer,
            train_freq=train_freq,
            grad_step=grad_step,
            weight_decay=1e-6,
            lr=lr,
            batch_size=batch_size,
        )
        model.learn(10000, reward_modifier=surprise_motivation, callback=learner)
        results.append(model.replay_buffer.nb_cells)

    return np.median(results)


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///example.db", direction="maximize", study_name="PandaReachSimHashexplore", load_if_exists=True
    )
    study.optimize(objective, n_trials=10)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
