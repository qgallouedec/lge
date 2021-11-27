import gym
import numpy as np
import optuna
import panda_gym
from go_explore.RND import RND, PredictorLearner
from go_explore.wrapper import UnGoalWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# Define an objective function to be minimized.
def objective(trial: optuna.Study):
    scaling_factor = trial.suggest_categorical("scaling_factor", [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256, 512])
    train_freq = trial.suggest_categorical("train_freq", [1, 5, 10, 50, 100, 500, 1000])
    grad_step = trial.suggest_categorical("grad_step", [1, 5, 10, 50, 100, 500, 1000])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    lr = trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    out_dim = trial.suggest_categorical("out_dim", [16, 32, 64, 128, 256, 512])

    rewards = []
    for _ in range(3):
        env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_reward=100)

        rnd = RND(
            scaling_factor=scaling_factor, obs_dim=env.observation_space.shape[0], out_dim=out_dim, hidden_dim=hidden_size
        )

        model = SAC("MlpPolicy", env, reward_modifier=rnd, verbose=1)
        cb = PredictorLearner(
            predictor=rnd.predictor,
            target=rnd.target,
            buffer=model.replay_buffer,
            train_freq=train_freq,
            grad_step=grad_step,
            weight_decay=weight_decay,
            lr=lr,
            batch_size=batch_size,
        )

        model.learn(8000, callback=cb)

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
        storage="sqlite:///example.db", study_name="PandaReachSACRND", direction="maximize", load_if_exists=True
    )
    study.optimize(objective, n_trials=10)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
