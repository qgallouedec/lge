import os

import gym
import gym_continuous_maze
import numpy as np
from stable_baselines3 import SAC
from toolbox.maze_grid import compute_coverage

from lge.lge import LatentGoExplore

NUM_TIMESTEPS = 30_000
NUM_RUN = 5

for run_idx in range(NUM_RUN):
    env = gym.make("ContinuousMaze-v0")
    model = LatentGoExplore(SAC, env, distance_threshold=0.4, p=0.01, latent_size=4, verbose=1)
    model.explore(NUM_TIMESTEPS)
    buffer = model.archive
    observations = buffer.next_observations["observation"][: buffer.pos if not buffer.full else buffer.buffer_size]
    coverage = compute_coverage(observations) / (24 * 24) * 100
    coverage = np.expand_dims(coverage, 0)

    filename = "results/lge_maze.npy"
    if os.path.exists(filename):
        previous_coverage = np.load(filename)
        counts = np.concatenate((previous_coverage, coverage))
    np.save(filename, coverage)
