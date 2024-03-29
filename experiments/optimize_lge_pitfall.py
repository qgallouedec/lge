# pip install ale-py==0.7.4
import argparse
import time

from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

import wandb
from experiments.utils import AtariWrapper, MaxRewardLogger
from lge import LatentGoExplore

NUM_TIMESTEPS = 500_000

parser = argparse.ArgumentParser()
parser.add_argument("--module_type", type=str)
parser.add_argument("--latent_size", type=int)
parser.add_argument("--distance_threshold", type=float)
parser.add_argument("--lighten_dist_coef", type=int)
parser.add_argument("--p", type=float)
parser.add_argument("--module_train_freq", type=int)
parser.add_argument("--module_grad_steps", type=int)


args = parser.parse_args()

env_id = "Pitfall-v4"
module_type = args.module_type
latent_size = args.latent_size
distance_threshold = args.distance_threshold
lighten_dist_coef = args.lighten_dist_coef
learning_starts = 100_000
p = args.p
n_envs = 4
nb_random_exploration_steps = 200
module_train_freq = args.module_train_freq
module_grad_steps = args.module_grad_steps


class NumberRoomsLogger(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.unique_rooms = set()

    def _on_step(self) -> bool:
        buffer = self.locals["replay_buffer"]  # type: ReplayBuffer
        infos = buffer.infos
        if not buffer.full:
            infos = buffer.infos[: buffer.pos]
        rooms = [info[env_idx]["ram"][1] for info in infos for env_idx in range(buffer.n_envs)]
        unique_rooms = set(rooms)
        self.unique_rooms = self.unique_rooms.union(unique_rooms)
        self.logger.record("env/explored_rooms", len(self.unique_rooms))
        return True


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
    model_kwargs=dict(buffer_size=200_000, policy_kwargs=dict(categorical=True)),
    wrapper_cls=AtariWrapper,
    nb_random_exploration_steps=nb_random_exploration_steps,
    module_train_freq=module_train_freq,
    module_grad_steps=module_grad_steps,
    tensorboard_log=f"runs/{run.id}",
    verbose=1,
)

model.explore(NUM_TIMESTEPS, CallbackList([NumberRoomsLogger(), MaxRewardLogger()]))
run.finish()
