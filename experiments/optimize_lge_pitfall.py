# pip install ale-py==0.7.4
import argparse

from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from wandb.integration.sb3 import WandbCallback

import wandb
from experiments.utils import MaxRewardLogger, RAMtoInfoWrapper
from lge import LatentGoExplore

NUM_TIMESTEPS = 500_000

parser = argparse.ArgumentParser()
parser.add_argument("--distance_threshold", type=float)
parser.add_argument("--latent_size", type=int)
parser.add_argument("--lighten_dist_coef", type=int)
parser.add_argument("--module_type", type=str)
parser.add_argument("--p", type=float)


args = parser.parse_args()

env_id = "Pitfall-v4"
module_type = args.module_type
latent_size = args.latent_size
distance_threshold = args.distance_threshold
lighten_dist_coef = args.lighten_dist_coef
learning_starts = 10_000
p = args.p
n_envs = 4


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
        self.logger.record("env/explored rooms", len(self.unique_rooms))
        return True


run = wandb.init(
    name=f"lge__{env_id}__{module_type}",
    project="lge",
    config=dict(
        env_id=env_id,
        module_type=module_type,
        latent_size=latent_size,
        distance_threshold=distance_threshold,
        lighten_dist_coef=lighten_dist_coef,
        leanring_starts=learning_starts,
        p=p,
        n_envs=n_envs,
    ),
    sync_tensorboard=True,
)

model = LatentGoExplore(
    DQN,
    env_id,
    module_type,
    latent_size,
    distance_threshold,
    lighten_dist_coef,
    p,
    n_envs,
    learning_starts=learning_starts,
    model_kwargs=dict(buffer_size=200_000, policy_kwargs=dict(categorical=True)),
    wrapper_cls=RAMtoInfoWrapper,
    tensorboard_log=f"runs/{run.id}",
    verbose=1,
)
wandb_callback = WandbCallback(
    gradient_save_freq=100,
    model_save_path=f"models/{run.id}",
    verbose=2,
)

model.explore(NUM_TIMESTEPS, CallbackList([NumberRoomsLogger(), MaxRewardLogger(), wandb_callback]))
run.finish()
