import go_explore.envs
import gym
import numpy as np
import optuna
from go_explore.simhash import SimHashMotivation
from stable_baselines3 import SAC


def objective(trial: optuna.Study):
    granularity = trial.suggest_categorical("granularity", [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    beta = trial.suggest_loguniform("beta", 1e-3, 1e3)

    rewards = []
    for _ in range(3):
        env = gym.make("PandaReachFlat-v0")

        simhash = SimHashMotivation(obs_dim=env.observation_space.shape[0], granularity=granularity, beta=beta)
        model = SAC("MlpPolicy", env, reward_modifier=simhash, verbose=1)
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
        storage="sqlite:///example.db", study_name="PandaReachSimHash", direction="maximize", load_if_exists=True
    )
    study.optimize(objective, n_trials=10)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
