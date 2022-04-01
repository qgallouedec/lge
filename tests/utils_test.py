import numpy as np

from go_explore.utils import index, indexes, multinomial, sample_geometric


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


def test_multinomial():
    weights = np.array([1, 2, 3])
    sample = [multinomial(weights) for _ in range(1000)]
    _, counts = np.unique(sample, return_counts=True)
    sampled_dist = counts / counts.sum()
    true_dist = weights / weights.sum()
    # If multinomial implementation is good, you have one chance in 350 that the test fails.
    assert np.isclose(sampled_dist, true_dist, atol=0.05).all()


def test_sample_geometric():
    mean = 3
    max_value = 6
    sample = [sample_geometric(mean, max_value) for _ in range(1000)]
    _, counts = np.unique(sample, return_counts=True)
    sampled_dist = counts / counts.sum()
    p = 1 / mean
    true_weights = np.array([(1 - p) ** (k - 1) * p for k in range(1, max_value)])
    true_dist = true_weights / true_weights.sum()
    # If multinomial implementation is good, you have one chance in 750 that the test fails.
    assert np.isclose(sampled_dist, true_dist, atol=0.05).all()
