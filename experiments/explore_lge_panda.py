import os

import gym
import numpy as np
import panda_gym
from stable_baselines3 import DDPG
from toolbox.panda_utils import compute_coverage

from lge import LatentGoExplore

NUM_TIMESTEPS = 1_000_000
NUM_RUN = 1

for run_idx in range(NUM_RUN):
    env = gym.make("PandaNoTask-v0", nb_objects=1)
    model = LatentGoExplore(
        DDPG,
        env,
        module_type="forward",
        latent_size=8,
        distance_threshold=1.0,
        lighten_dist_coef=0.0,
        p=0.001,
        model_kwargs=dict(buffer_size=NUM_TIMESTEPS),
        verbose=1,
    )
    model.explore(NUM_TIMESTEPS)
    buffer = model.replay_buffer
    observations = buffer.next_observations["observation"][: buffer.pos if not buffer.full else buffer.buffer_size]
    coverage = compute_coverage(observations)
    coverage = np.expand_dims(coverage, 0)

    filename = "results/lge_panda.npy"
    if os.path.exists(filename):
        previous_coverage = np.load(filename)
        coverage = np.concatenate((previous_coverage, coverage))
    np.save(filename, coverage)
