import gym
import numpy as np
import optuna
import panda_gym
from go_explore.simhash import SimHashWrapper
from go_explore.wrapper import UnGoalWrapper
from stable_baselines3 import TD3
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


# Define an objective function to be minimized.
def objective(trial: optuna.Study):
    beta = trial.suggest_loguniform("beta", 1e-2, 1e2)
    granularity = trial.suggest_categorical("granularity", [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])

    env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_reward=100)
    env = SimHashWrapper(env, granularity=granularity, beta=beta)

    action_noise_cls = OrnsteinUhlenbeckActionNoise
    action_noise = action_noise_cls(mean=np.zeros(env.action_space.shape), sigma=np.ones(env.action_space.shape) * 0.5)

    rewards = []
    for _ in range(1):
        model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
        model.learn(30000)

        eval_env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_reward=100)
        sum_reward = 0
        for _ in range(50):
            obs = eval_env.reset()
            done = False
            while not done:
                action = model.predict(obs)[0]
                obs, reward, done, info = eval_env.step(action)
                sum_reward += reward
        rewards.append(sum_reward)
    return np.median(rewards)


if __name__ == "__main__":
    import optuna.visualization

    study = optuna.create_study(
        storage="sqlite:///example.db", study_name="PandaReachSimHash", direction="maximize", load_if_exists=True
    )
    study.optimize(objective, n_trials=10)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
