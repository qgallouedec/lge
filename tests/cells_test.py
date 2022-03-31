import gym
import numpy as np
import torch as th
from go_explore.cells import ImageGrayscaleDownscale

# Produce images
env = gym.make("MontezumaRevenge-v0")
images = [env.reset()]
for _ in range(300):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    images.append(obs)
images = th.from_numpy(np.array(images)).moveaxis(-1, -3)


def test_get_cells():
    cell_factory = ImageGrayscaleDownscale(30, 40)
    cells = cell_factory(images)
    assert cells.shape == (301, 30 * 40)


def test_optimization():
    cell_factory = ImageGrayscaleDownscale()
    # Optimize parameters
    cell_factory.optimize_param(images)
    # Get all the unique cells. It should produced around 38 cells (301*0.125)
    cells = th.unique(cell_factory(images), dim=0)
    # It should produce at least one cell (probably around 38 in fact)
    assert cells.shape[0] > 1
