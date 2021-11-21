import gym
import numpy as np
import optuna

from stable_baselines3 import TD3, HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# Define an objective function to be minimized.
def objective(trial: optuna.Study):
    env = gym.make("MountainCarContinuous-v0")

    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024])
    tau = trial.suggest_categorical("tau", [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    train_freq = trial.suggest_categorical("train_freq", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    net_depth = trial.suggest_categorical("net_depth", [2, 3, 4])
    net_width = trial.suggest_categorical("net_width", [32, 64, 128, 256])
    net_arch = [net_width for _ in range(net_depth)]

    action_noise_cls = trial.suggest_categorical("action_noise_cls", ["Normal", "OrnsteinUhlenbeck", "None"])
    action_noise_cls = {"Normal": NormalActionNoise, "OrnsteinUhlenbeck": OrnsteinUhlenbeckActionNoise, "None": None}[
        action_noise_cls
    ]
    action_noise_sigma = trial.suggest_loguniform("action_noise_std", 1e-5, 1)
    if action_noise_cls is not None:
        action_noise = action_noise_cls(
            mean=np.zeros(env.action_space.shape), sigma=np.ones(env.action_space.shape) * action_noise_sigma
        )
    else:
        action_noise = None
    policy_delay = trial.suggest_categorical("policy_delay", [1, 2, 3, 5, 7, 10])
    target_policy_noise = trial.suggest_loguniform("target_policy_noise", 1e-5, 1)
    target_noise_clip = trial.suggest_loguniform("target_noise_clip", 1e-5, 1)

    for _ in range(3):
        model = TD3(
            "MlpPolicy",
            env,
            gamma=gamma,
            learning_rate=learning_rate,
            batch_size=batch_size,
            tau=tau,
            train_freq=train_freq,
            action_noise=action_noise,
            policy_kwargs=dict(net_arch=net_arch),
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            verbose=0,
        )
        model.learn(300000)

        rewards = 0
        for _ in range(50):
            obs = env.reset()
            done = False
            while not done:
                action = model.predict(obs)[0]
                obs, reward, done, info = env.step(action)
                rewards += reward

    return rewards


if __name__ == "__main__":
    import optuna.visualization

    study = optuna.create_study(
        storage="sqlite:///example.db", study_name="MountainCarContinuous300k", direction="maximize", load_if_exists=True
    )
    study.optimize(objective, n_trials=10)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
