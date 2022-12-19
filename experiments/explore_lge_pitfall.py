# pip install ale-py==0.7.4
import time

from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from wandb.integration.sb3 import WandbCallback

import wandb
from experiments.utils import AtariWrapper, MaxRewardLogger
from lge import LatentGoExplore

NUM_TIMESTEPS = 100_000
NUM_RUN = 1

env_id = "Pitfall-v4"
module_type = "forward"
latent_size = 32
distance_threshold = 1.0
lighten_dist_coef = 1.0
learning_starts = 100_000
p = 0.1
n_envs = 4
nb_random_exploration_steps = 200
module_train_freq = 10_000
module_grad_steps = 500


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
    model_kwargs=dict(buffer_size=200_000, policy_kwargs=dict(categorical=True), exploration_fraction=0.5),
    wrapper_cls=AtariWrapper,
    nb_random_exploration_steps=nb_random_exploration_steps,
    module_train_freq=module_train_freq,
    module_grad_steps=module_grad_steps,
    tensorboard_log=f"runs/{run.id}",
    verbose=1,
)


model.explore(NUM_TIMESTEPS, CallbackList([NumberRoomsLogger(), MaxRewardLogger()]))
run.finish()
