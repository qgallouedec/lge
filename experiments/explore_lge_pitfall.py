# pip install ale-py==0.7.4
import argparse
import time

from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

import wandb
from experiments.utils import AtariWrapper, MaxRewardLogger, NumberCellsLogger
from lge import LatentGoExplore

NUM_TIMESTEPS = 1_000_000

parser = argparse.ArgumentParser()
parser.add_argument("--module_type", type=str, default="ae")
parser.add_argument("--latent_size", type=int, default=32)
parser.add_argument("--distance_threshold", type=float, defaul=0.1)
parser.add_argument("--lighten_dist_coef", type=float, default=1.0)
parser.add_argument("--p", type=float, default=0.1)
parser.add_argument("--module_train_freq", type=int, default=10_000)
parser.add_argument("--module_grad_steps", type=int, default=500)
parser.add_argument("--n-envs", type=int, default=8)


args = parser.parse_args()

env_id = "Pitfall-v4"
module_type = args.module_type
latent_size = args.latent_size
distance_threshold = args.distance_threshold
lighten_dist_coef = args.lighten_dist_coef
learning_starts = 100_000
p = args.p
n_envs = args.n_envs
nb_random_exploration_steps = 200
module_train_freq = args.module_train_freq
module_grad_steps = args.module_grad_steps


run = wandb.init(
    name=f"lge__{env_id}__{module_type}__{str(time.time())}",
    project="lge",
    config=dict(
        env_id=env_id,
        module_type=module_type,
        latent_size=latent_size,
        distance_threshold=distance_threshold,
        lighten_dist_coef=lighten_dist_coef,
        p=p,
        n_envs=n_envs,
        learning_starts=learning_starts,
        nb_random_exploration_steps=nb_random_exploration_steps,
        module_train_freq=module_train_freq,
        module_grad_steps=module_grad_steps,
    ),
    sync_tensorboard=True,
)

model = LatentGoExplore(
    DQN,
    env_id,
    module_type=module_type,
    latent_size=latent_size,
    distance_threshold=distance_threshold,
    lighten_dist_coef=lighten_dist_coef,
    p=p,
    n_envs=n_envs,
    learning_starts=learning_starts,
    model_kwargs=dict(buffer_size=n_envs * 400_000, policy_kwargs=dict(categorical=True), exploration_fraction=0.5),
    wrapper_cls=AtariWrapper,
    nb_random_exploration_steps=nb_random_exploration_steps,
    module_train_freq=module_train_freq,
    module_grad_steps=module_grad_steps,
    tensorboard_log=f"runs/{run.id}",
    verbose=1,
)


model.explore(NUM_TIMESTEPS, CallbackList([MaxRewardLogger(), NumberCellsLogger()]))
run.finish()
