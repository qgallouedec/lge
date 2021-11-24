import gym
import numpy as np
import optuna
from go_explore.surprise import SurpriseWrapper, TransitionModel, TransitionModelLearner
from go_explore.wrapper import StoreTransitionsWrapper, UnGoalWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


# Define an objective function to be minimized.
def objective(trial: optuna.Study):
    eta = trial.suggest_loguniform("eta", 1e-3, 1e2)

    rewards = []
    for _ in range(3):
        env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_reward=100)
        env = StoreTransitionsWrapper(env)
        transition_model = TransitionModel(env.observation_space.shape[0], env.action_space.shape[0])
        env = SurpriseWrapper(env, transition_model, eta=eta)

        action_noise_cls = OrnsteinUhlenbeckActionNoise
        action_noise = action_noise_cls(mean=np.zeros(env.action_space.shape), sigma=np.ones(env.action_space.shape) * 0.5)

        model = SAC("MlpPolicy", env, action_noise=action_noise, verbose=1)
        cb = TransitionModelLearner(transition_model=transition_model, buffer=env.replay_buffer, train_freq=10)

        model.learn(20000, callback=cb)

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
    study = optuna.create_study(
        storage="sqlite:///example.db", study_name="PandaReachSACSurprise", direction="maximize", load_if_exists=True
    )
    study.optimize(objective, n_trials=10)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
