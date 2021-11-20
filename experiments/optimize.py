import gym
import numpy as np
import optuna

from stable_baselines3 import TD3, HerReplayBuffer

# Define an objective function to be minimized.
def objective(trial: optuna.Study):
    env = gym.make("MountainCarContinuous-v0")

    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024])
    tau = trial.suggest_categorical("tau", [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    train_freq = trial.suggest_categorical("train_freq", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    net_arch = tuple([[nb_neur for _ in range(depth)] for nb_neur in [32, 64, 128, 256] for depth in [2, 3]])
    net_arch = trial.suggest_categorical("net_arch", net_arch)

    for _ in range(3):
        model = TD3(
            "MlpPolicy",
            env,
            gamma=gamma,
            learning_rate=learning_rate,
            batch_size=batch_size,
            tau=tau,
            train_freq=train_freq,
            policy_kwargs={"net_arch": net_arch},
            verbose=1,
        )
        model.learn(100000)

        rewards = 0
        for _ in range(50):
            obs = env.reset()
            done=False
            while not done:
                action = model.predict(obs)[0]
                obs, reward, done, info = env.step(action)
                rewards += reward
    
    return rewards


if __name__ == "__main__":
    import optuna.visualization
    study = optuna.create_study(
        storage="sqlite:///example.db", study_name="MountainCarContinuous", direction="maximize", load_if_exists=True
    )
    study.optimize(objective, n_trials=10)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
