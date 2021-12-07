import go_explore.envs
import gym
import numpy as np
import optuna
from go_explore.icm import ICM
from stable_baselines3 import SAC


def objective(trial: optuna.Study):
    beta = trial.suggest_categorical("beta", [0, 0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 0.999, 1])
    scaling_factor = trial.suggest_loguniform("scaling_factor", 1e-3, 1e3)
    lmbda = trial.suggest_loguniform("lmbda", 1e-3, 1e3)
    feature_dim = trial.suggest_categorical("feature_dim", [8, 16, 32, 64, 128, 256])
    hidden_dim = trial.suggest_categorical("hidden_dim", [8, 16, 32, 64, 128, 256])

    rewards = []
    for _ in range(3):
        env = gym.make("PandaReachFlat-v0")
        icm = ICM(
            beta=beta,
            scaling_factor=scaling_factor,
            lmbda=lmbda,
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
        )
        model = SAC("MlpPolicy", env, actor_loss_modifier=icm, reward_modifier=icm, verbose=1)
        model.learn(8000)

        sum_reward = 0
        for _ in range(50):
            obs = env.reset()
            done = False
            while not done:
                action = model.predict(obs)[0]
                obs, reward, done, info = env.step(action)
                sum_reward += reward
        rewards.append(sum_reward)
    return np.median(rewards)


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///example.db", study_name="PandaReachICM", direction="maximize", load_if_exists=True
    )
    study.optimize(objective, n_trials=10)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
