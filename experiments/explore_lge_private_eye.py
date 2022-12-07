from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

from lge import LatentGoExplore

NUM_TIMESTEPS = 100_000
NUM_RUN = 1


model = LatentGoExplore(
    DQN,
    "PrivateEye-v4",
    module_type="ae",
    latent_size=32,
    distance_threshold=1.0,
    lighten_dist_coef=1.0,
    p=0.05,
    model_kwargs=dict(buffer_size=NUM_TIMESTEPS),
    wrapper_cls=AtariWrapper,
    verbose=1,
)
model.explore(NUM_TIMESTEPS)
