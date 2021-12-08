import numpy as np
import optuna
from go_explore.envs import PandaReachFlat
from go_explore.surprise import SurpriseMotivation, TransitionModelLearner
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# Define an objective function to be minimized.
def objective(trial: optuna.Study):
    eta = trial.suggest_loguniform("eta", 1e-3, 1e6)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256, 512])
    train_freq = trial.suggest_categorical("train_freq", [1, 5, 10, 50, 100, 500, 1000])
    grad_step = trial.suggest_categorical("grad_step", [1, 5, 10, 50, 100, 500, 1000])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-7, 1e-2)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])

    rewards = []
    for _ in range(3):
        env = DummyVecEnv([PandaReachFlat])
        env = VecNormalize(env, norm_reward=False)
        surprise_motivation = SurpriseMotivation(
            obs_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], eta=eta, hidden_size=hidden_size
        )
        model = SAC("MlpPolicy", env, reward_modifier=surprise_motivation, verbose=1)
        cb = TransitionModelLearner(
            transition_model=surprise_motivation.transition_model,
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
        storage="sqlite:///example.db", study_name="PandaReachSACSurprise", direction="maximize", load_if_exists=True
    )
    study.optimize(objective, n_trials=10)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
