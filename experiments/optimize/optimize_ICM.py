import numpy as np
import optuna
from go_explore.envs import PandaReachFlat
from go_explore.icm import ICM
from optuna.pruners import BasePruner
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class MyPruner(BasePruner):
    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        try:
            best = max(study.best_trial.intermediate_values.keys())
        except ValueError:
            print("not best yet")
            return False
        current = max(trial.intermediate_values.keys())
        if current > best:
            return True
        else:
            return False


def objective(trial: optuna.Study):
    beta = trial.suggest_categorical("beta", [0, 0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 0.999, 1])
    scaling_factor = trial.suggest_loguniform("scaling_factor", 1e-3, 1e3)
    lmbda = trial.suggest_loguniform("lmbda", 1e-3, 1e3)
    feature_dim = trial.suggest_categorical("feature_dim", [8, 16, 32, 64, 128, 256])
    hidden_dim = trial.suggest_categorical("hidden_dim", [8, 16, 32, 64, 128, 256])

    env = DummyVecEnv([PandaReachFlat])
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
    models = [SAC("MlpPolicy", env, actor_loss_modifier=icm, reward_modifier=icm, verbose=1) for _ in range(3)]

    for _ in range(20):
        for model in models:
            model.learn(500, reset_num_timesteps=False)

        if models[0].num_timesteps < 6000:
            continue

        # evaluate
        rewards = []
        for model in models:
            sum_reward = 0
            for _ in range(50):
                obs = env.reset()
                done = False
                while not done:
                    action = model.predict(obs)[0]
                    obs, reward, done, info = env.step(action)
                    sum_reward += reward
            rewards.append(sum_reward)

        # report
        trial.report(np.median(rewards), models[0].num_timesteps)

        # prune, if necessary
        if trial.should_prune():
            raise optuna.TrialPruned()

        if np.median(rewards) > -1000:
            return models[0].num_timesteps

    return models[0].num_timesteps  # not learned


if __name__ == "__main__":
    pruner = MyPruner()
    study = optuna.create_study(storage="sqlite:///example.db", study_name="PandaReachICM", load_if_exists=True, pruner=pruner)
    study.optimize(objective, n_trials=10)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
