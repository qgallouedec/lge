from typing import Optional, Tuple, Union

import numpy as np
import torch
from gym import spaces
from torch import Tensor


def indexes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Indexes of a in b.

    Args:
        a (np.ndarray): Array of shape (...)
        b (np.ndarray): Array of shape (N x ...)

    Returns:
        np.ndarray: Indexes of the occurences of a in b
    """
    if b.shape[0] == 0:
        return np.array([])
    a = a.flatten()
    b = b.reshape((b.shape[0], -1))
    idxs = np.where((a == b).all(1))[0]
    return idxs


def index(a: np.ndarray, b: np.ndarray) -> Optional[int]:
    """
    Index of first occurence of a in b.

    Args:
        a (np.ndarray): Array of shape (...)
        b (np.ndarray): Array of shape (N x ...)

    Returns:
        np.ndarray: index of the first occurence of a in b
    """
    idxs = indexes(a, b)
    if idxs.shape[0] == 0:
        return None
    else:
        return idxs[0]


def sample_geometric_with_max(
    p: float, max_value: int, size: Optional[Union[int, Tuple[int]]] = None
) -> Union[int, np.ndarray]:
    """
    Sample follow geometric law, but are below the max_value.

    Args:
        p (int): The probability of success of an individual trial
        max_value (int): Maximum value for the sample, included
        size (Optional[Union[int, Tuple[int]]]): Output size. If None, sample a single int

    Returns:
        Union[int, np.ndarray]: Sampled value
    """
    if p > 0:
        for _ in range(10_000):
            sample = np.random.geometric(p, size)
            if np.all(sample <= max_value):
                return sample
    return np.random.randint(0, max_value + 1, size)


def is_image(x: Tensor) -> bool:
    """
    Whether the input is an image, or a batch of images.

    Args:
        x (Tensor): Input Tensor

    Returns:
        bool: Whether it's an image
    """

    shape = x.shape

    if len(shape) >= 3 and 3 in shape:
        return True
    else:
        return False


def round(input: Tensor, decimals: float) -> Tensor:
    """
    Rounding, but extended to every float.

    :param input: Input tensor
    :param decimals: Decimals, can be float
    :return: The rounded tensor

    Example:
    >>> a
    tensor([0.0000, 0.4000, 0.8000, 1.2000, 1.6000])
    >>> torch.round(a, decimals=0.8)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: round() received an invalid combination of arguments - got (Tensor, decimals=float), but expected one of:
     * (Tensor input, *, Tensor out)
     * (Tensor input, *, int decimals, Tensor out)
    >>> round(a, decimals=0.2)
    tensor([0.0000, 0.6310, 0.6310, 1.2619, 1.8929])
    """
    return torch.round(input * 10**decimals) / 10**decimals


def estimate_density(x: Tensor, samples: Tensor) -> Tensor:
    """
    Estimate the density of x within the dataset

    :param x: Points to evaluate density
    :param dataset: The samples from the distribution to estimate
    :return: The estiamte density on x
    """
    n, d = samples.shape
    k = int(2 * n ** (1 / d))
    cdist = torch.cdist(x, samples)
    dist_to_kst = cdist.topk(k, largest=False)[0][:, -1]
    return dist_to_kst ** (-d)


def lighten(arr: np.ndarray, threshold: float) -> np.ndarray:
    """
    Returns the indexes of the input array such that all successive elements are
    at least distant from the threshold value. It keeps the last one in place.

    Example:
    >>> arr = np.array([4.0, 5.0, 5.1, 6.0, 7.0])
    >>> idxs = lighten(arr, threshold=1.0)
    >>> idxs
    array([0, 1, 3, 4])
    >>> arr[idxs]
    array([4., 5., 6., 7.])

    :param arr: Input array
    :param threshold: Distance threshold
    :return: List of indexes
    """
    arr = arr[::-1]  # flip array
    idxs = np.arange(len(arr))[::-1]  # [..., 2, 1, 0]
    idx = 0
    while idx + 1 < len(arr):
        dist = np.linalg.norm(arr[idx] - arr[idx + 1])
        if dist < threshold:
            arr = np.delete(arr, idx + 1, 0)
            idxs = np.delete(idxs, idx + 1, 0)
        else:
            idx += 1
    return idxs[::-1]  # reflip array


def get_size(space: spaces.Space) -> int:
    """
    Get the dimension of the space when flattened.

    :param space: Space
    :return: The size
    """
    if isinstance(space, spaces.Discrete):
        return space.n
    elif isinstance(space, spaces.MultiDiscrete):
        return sum(space.nvec)
    elif isinstance(space, spaces.MultiBinary):
        return space.n if isinstance(space.n, int) else sum(space.n)
    elif isinstance(space, spaces.Box):
        return np.prod(space.shape)
    else:
        raise ValueError
