import gym
import panda_gym
from go_explore.cell_computers import PandaObjectCellComputer
from go_explore.go_explore import GoExplore
from go_explore.wrapper import UnGoalWrapper
from stable_baselines3 import HerReplayBuffer

env = gym.make("PandaPickAndPlace-v1", render=True)
env = UnGoalWrapper(env)
cell_computer = PandaObjectCellComputer()

go_explore = GoExplore(
    env=env,
    cell_computer=cell_computer,
    go_timesteps=10000,
    explore_timesteps=0,
    horizon=10,
    count_pow=4,
    learning_rate=0.001057,
    batch_size=1024,
    tau=0.02,
    gamma=0.995,
    train_freq=32,
    policy_kwargs=dict(net_arch=[128, 128, 128]),
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs={
        "n_sampled_goal": 5,
        "goal_selection_strategy": "future",
        "online_sampling": True,
        "max_episode_length": 50,
    },
    verbose=1,
)
go_explore.exploration(20000)
