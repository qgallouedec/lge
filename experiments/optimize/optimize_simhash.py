import numpy as np
import optuna
from go_explore.envs import PandaReachFlat
from go_explore.simhash import SimHashMotivation
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def objective(trial: optuna.Study):
    granularity = trial.suggest_categorical("granularity", [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    beta = trial.suggest_loguniform("beta", 1e-3, 1e3)

    rewards = []
    for _ in range(3):
        env = DummyVecEnv([PandaReachFlat])
        env = VecNormalize(env, norm_reward=False)
        model = SAC("MlpPolicy", env, verbose=1)
        simhash = SimHashMotivation(model.replay_buffer, env, granularity=granularity, beta=beta)
        model.learn(8000, reward_modifier=simhash)

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
