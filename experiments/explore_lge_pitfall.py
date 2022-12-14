# pip install ale-py==0.7.4
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from wandb.integration.sb3 import WandbCallback

import wandb
from experiments.utils import MaxRewardLogger, RAMtoInfoWrapper
from lge import LatentGoExplore


NUM_TIMESTEPS = 100_000
NUM_RUN = 1

env_id = "Pitfall-v4"
module_type = "ae"
latent_size = 32
distance_threshold = 1.0
lighten_dist_coef = 1.0
p = 0.05
n_envs = 2


class NumberRoomsLogger(BaseCallback):
    def _on_step(self) -> bool:
        buffer = self.locals["replay_buffer"]  # type: ReplayBuffer
        infos = buffer.infos
        if not buffer.full:
            infos = buffer.infos[: buffer.pos]
        rooms = [info[env_idx]["ram"][1] for info in infos for env_idx in range(buffer.n_envs)]
        unique_rooms = np.unique(rooms)
        self.logger.record("env/explored rooms", len(unique_rooms))
        return True


run = wandb.init(
    project="lge",
    config=dict(
        env_id=env_id,
        module_type=module_type,
        latent_size=latent_size,
        distance_threshold=distance_threshold,
        lighten_dist_coef=lighten_dist_coef,
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
    model_kwargs=dict(buffer_size=100_000, policy_kwargs=dict(categorical=True)),
    wrapper_cls=RAMtoInfoWrapper,
    tensorboard_log=f"runs/{run.id}",
    verbose=1,
)

model.explore(NUM_TIMESTEPS, CallbackList([NumberRoomsLogger(), MaxRewardLogger(), WandbCallback()]))
run.finish()
