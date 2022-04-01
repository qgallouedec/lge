from math import isclose

import gym
import numpy as np
import torch as th
from gym import spaces

from go_explore.cells import CellIsObs, DownscaleObs, ImageGrayscaleDownscale, distribution_score, get_param_score

# Produce images
env = gym.make("MontezumaRevenge-v0")
images = [env.reset()]
for _ in range(300):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    images.append(obs)
images = th.from_numpy(np.array(images)).moveaxis(-1, -3)


def test_distribution_score1():
    probs = th.Tensor([0.2, 0.2, 0.2, 0.2, 0.2])
    nb_samples = 10
    split_factor = 0.125
    score = distribution_score(probs, nb_samples, split_factor)
    assert isclose(score, 0.5, abs_tol=0.000001)


def test_distribution_score2():
    probs = th.Tensor([0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05])
    nb_samples = 30
    split_factor = 0.3
    score = distribution_score(probs, nb_samples, split_factor)
    assert isclose(score, 0.820467, abs_tol=0.000001)


def test_get_param_score():
    cells = th.Tensor(
        [
            [0.00, 0.00],  # Cell 1
            [0.00, 0.00],  # Cell 1
            [1.00, 0.00],  # Cell 2
            [1.00, 0.00],  # Cell 2
            [1.00, 2.00],  # Cell 3
            [1.00, 0.00],  # Cell 2
            [1.00, 0.00],  # Cell 2
            [1.01, 0.00],  # Cell 4
            [1.01, 0.00],  # Cell 4
            [1.01, 0.00],  # Cell 4
            [1.01, 0.00],  # Cell 4
        ]
    )
    score = get_param_score(cells, split_factor=0.125)
    assert isclose(score, 0.534434, abs_tol=0.000001)


def test_image_cell():
    cell_factory = ImageGrayscaleDownscale(30, 40)
    cells = cell_factory(images)
    assert cells.shape == (301, 30 * 40)


def test_transpose_image_cell():
    cell_factory = ImageGrayscaleDownscale(30, 40)
    transposed_images = images.moveaxis(-3, -1)  # (... x 3 x H x W) to (... x H x W x 3)
    cells_1 = cell_factory(images)
    cells_2 = cell_factory(transposed_images)
    assert (cells_1 == cells_2).all()


def test_image_grayscale_downscale_optimization():
    cell_factory = ImageGrayscaleDownscale()
    # Optimize parameters
    cell_factory.optimize_param(images)
    # Get all the unique cells. It should produced around 38 cells (301*0.125)
    cells = th.unique(cell_factory(images), dim=0)
    # It should produce at least one cell (probably around 38 in fact)
    assert cells.shape[0] > 1


def test_image_grayscale_downscale_old_optimization():
    cell_factory = ImageGrayscaleDownscale()
    # Optimize parameters
    cell_factory.old_optimize_param(images)
    # Get all the unique cells. It should produced around 38 cells (301*0.125)
    cells = th.unique(cell_factory(images), dim=0)
    # It should produce at least one cell (probably around 38 in fact)
    assert cells.shape[0] > 1


def test_cell_is_obs():
    cell_factory = CellIsObs(spaces.Box(-1, 1, (2,)))
    cell = cell_factory(th.Tensor([0.5, 0.5]))
    assert (cell == th.Tensor([0.5, 0.5])).all()


def test_cell_is_obs_optimization():
    # Nothing should append
    cell_factory = CellIsObs(spaces.Box(-1, 1, (2,)))
    cell_before = cell_factory(th.Tensor([0.5, 0.5]))
    cell_factory.optimize_param(th.Tensor([[0.5, 0.5]]))
    cell_after = cell_factory(th.Tensor([0.5, 0.5]))
    assert (cell_before == cell_after).all()


def test_downscale_obs():
    cell_factory = DownscaleObs(spaces.Box(-1, 1, (2,)))
    cell = cell_factory(th.Tensor([0.5, 0.5]))
    assert (cell == th.Tensor([0.0, 0.0])).all()


def test_downscale_obs_optimization():
    cell_factory = DownscaleObs(spaces.Box(-1, 1, (2,)))
    observations = th.tensor(
        [
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
            [0.5, 0.5],
            [0.6, 0.6],
            [0.7, 0.7],
            [0.8, 0.8],
            [0.9, 0.9],
            [1.0, 1.0],
        ]
    )
    cell_factory.optimize_param(observations)
    _, counts = th.unique(cell_factory(observations), dim=0, return_counts=True)
    # Best option is to split in two cells, with egal number of observations
    assert (counts == th.tensor([5, 5])).all()
