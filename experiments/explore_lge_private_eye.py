# pip install ale-py==0.7.4
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from experiments.utils import RAMtoInfoWrapper
from lge import LatentGoExplore

NUM_TIMESTEPS = 100_000
NUM_RUN = 1


class NumberRoomsLogger(BaseCallback):
    def _on_step(self) -> bool:
        buffer = self.locals["replay_buffer"]
        infos = buffer.infos
        if not buffer.full:
            infos = buffer.infos[: buffer.pos]
        rooms = [info[0]["ram"][92] for info in infos]
        unique_rooms = np.unique(rooms)
        self.logger.record("explored rooms", len(unique_rooms))
        return True


model = LatentGoExplore(
    DQN,
    "PrivateEye-v4",
    module_type="ae",
    latent_size=32,
    distance_threshold=1.0,
    lighten_dist_coef=1.0,
    p=0.05,
    model_kwargs=dict(buffer_size=NUM_TIMESTEPS),
    wrapper_cls=RAMtoInfoWrapper,
    verbose=1,
)
model.explore(NUM_TIMESTEPS, NumberRoomsLogger())
