import os

import gym
import gym_robotics
import numpy as np
from stable_baselines3 import SAC
from toolbox.fetch_utils import cumulative_object_coverage

from lge.lge import LatentGoExplore

NUM_TIMESTEPS = 1_000_000
NUM_RUN = 5

for run_idx in range(NUM_RUN):
    env = gym.make("__root__/FetchNoTask-v1")
    model = LatentGoExplore(SAC, env, distance_threshold=0.5, p=0.01, latent_size=8, verbose=1)
    model.explore(NUM_TIMESTEPS)
    buffer = model.archive
    observations = buffer.next_observations["observation"][: buffer.pos if not buffer.full else buffer.buffer_size]
    coverage = cumulative_object_coverage(observations)
    coverage = np.expand_dims(coverage, 0)

    filename = "results/lge_fetch.npy"
    if os.path.exists(filename):
        previous_coverage = np.load(filename)
        coverage = np.concatenate((previous_coverage, coverage))
    np.save(filename, coverage)
