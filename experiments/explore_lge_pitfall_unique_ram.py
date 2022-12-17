# pip install ale-py==0.7.4
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
learning_starts = 5_000
p = 0.1
n_envs = 4


class UniqueRAMLogger(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.unique_rams = set()

    def _on_step(self) -> bool:
        if self.num_timesteps % 1_000 == 0:
            buffer = self.locals["replay_buffer"]  # type: ReplayBuffer
            infos = buffer.infos
            if not buffer.full:
                infos = buffer.infos[: buffer.pos]
            rams = [info[env_idx]["ram"] for info in infos for env_idx in range(buffer.n_envs)]
            ram_hashes = [hash(ram.tobytes()) for ram in rams]
            unique_rams = set(ram_hashes)
            self.unique_rams = self.unique_rams.union(unique_rams)
            self.logger.record("env/unique_rams", len(self.unique_rams))
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
    model_kwargs=dict(buffer_size=100_000, policy_kwargs=dict(categorical=True)),
    wrapper_cls=RAMtoInfoWrapper,
    tensorboard_log=f"runs/{run.id}",
    verbose=1,
)
wandb_callback = WandbCallback(
    gradient_save_freq=100,
    model_save_path=f"models/{run.id}",
    verbose=2,
)

model.explore(NUM_TIMESTEPS, CallbackList([UniqueRAMLogger(), MaxRewardLogger(), wandb_callback]))
run.finish()
