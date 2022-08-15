import numpy as np
import torch
from torch.distributions import MultivariateNormal

from lge.utils import estimate_density, index, indexes, round, sample_geometric_with_max


def test_indexes():
    a = np.array([3, 4])
    b = np.array([[1, 2], [3, 5], [4, 3], [3, 4], [3, 4], [5, 4]])
    assert (indexes(a, b) == np.array([3, 4])).all()


def test_indexes_when_none():
    a = np.array([-1, -1])
    b = np.array([[1, 2], [3, 5], [4, 3], [3, 4], [3, 4], [5, 4]])
    assert (indexes(a, b) == np.array([])).all()


def test_index():
    a = np.array([3, 4])
    b = np.array([[1, 2], [3, 5], [4, 3], [3, 4], [3, 4]])
    assert index(a, b) == 3


def test_index_when_none():
    a = np.array([-1, -1])
    b = np.array([[1, 2], [3, 5], [4, 3], [3, 4], [3, 4]])
    assert index(a, b) is None


def test_sample_geometric_with_max():
    mean = 3
    p = 1 / mean
    max_value = 6
    sample = [sample_geometric_with_max(p, max_value) for _ in range(1000)]
    _, counts = np.unique(sample, return_counts=True)
    sampled_dist = counts / counts.sum()
    true_weights = np.array([(1 - p) ** (k - 1) * p for k in range(1, max_value + 1)])
    true_dist = true_weights / true_weights.sum()
    # If geometric implementation is good, you have one chance in 750 that the test fails.
    assert np.isclose(sampled_dist, true_dist, atol=0.05).all()


def test_round():
    x = torch.Tensor([0.0, 0.4, 0.8, 1.2, 1.6])
    y = round(x, decimals=0.2)
    z = torch.Tensor([0.0000, 0.6310, 0.6310, 1.2619, 1.8929])
    torch.isclose(y, z, atol=0.0001)
