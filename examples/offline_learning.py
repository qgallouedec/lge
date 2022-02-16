import gym
import panda_gym
from go_explore.common.wrappers import UnGoalWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
env = VecNormalize(env, norm_reward=False)

model1 = SAC("MlpPolicy", env, verbose=1)
model1.learn(7000, eval_env=env, eval_freq=2000, n_eval_episodes=10)

sum_reward = 0
for _ in range(50):
    obs = env.reset()
    done = False
    while not done:
        action = model1.predict(obs)[0]
        obs, reward, done, info = env.step(action)
        sum_reward += reward
print(sum_reward)

model2 = SAC("MlpPolicy", env, verbose=1)
model2.replay_buffer = model1.replay_buffer
model2._setup_learn(8000, env)
model2.train(8000)

sum_reward = 0
for _ in range(50):
    obs = env.reset()
    done = False
    while not done:
        action = model2.predict(obs)[0]
        obs, reward, done, info = env.step(action)
        sum_reward += reward
print(sum_reward)
