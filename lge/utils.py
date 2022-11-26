from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from stable_baselines3.common.preprocessing import is_image_space
from torch import Tensor


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


def get_shape(space: spaces.Space) -> Tuple[int]:
    """
    Get the shape of the space.

    :param space: Space
    :return: The size
    """
    if isinstance(space, spaces.Discrete):
        return (space.n,)
    elif isinstance(space, spaces.MultiDiscrete):
        return (sum(space.nvec),)
    elif isinstance(space, spaces.MultiBinary):
        return (space.n,) if isinstance(space.n, int) else (sum(space.n),)
    elif isinstance(space, spaces.Box):
        if is_image_space(space):
            if space.shape[2] in [1, 3]:  # channel last -> make channel first
                return (space.shape[2], *space.shape[:2])
            else:
                return space.shape
        else:
            return (np.prod(space.shape),)
    else:
        raise ValueError


def get_size(space: spaces.Space) -> int:
    """
    Get the dimension of the space when flattened.

    :param space: Space
    :return: The size
    """
    if is_image_space(space):
        raise Warning("Why are you computing the size of an image?")
    return np.prod(get_shape(space))


def is_batched(input: Tensor, space: spaces.Space):
    if isinstance(space, (spaces.Box, spaces.MultiDiscrete, spaces.MultiBinary)):
        if input.shape == space.shape:
            return False
        elif input.shape[1:] == space.shape:
            return True
        else:
            raise ValueError(f"Wrong input shape")
    elif isinstance(space, spaces.Discrete):
        if len(input.shape) == 0:
            return False
        elif len(input.shape) == 1:
            return True
        else:
            raise ValueError(f"Wrong input shape")
    else:
        raise ValueError(f"Space {space} not supported.")


def batchify(input: Union[Tensor, Dict[str, Tensor]]):
    if isinstance(input, dict):
        return {key: batchify(value) for key, value in input.items()}
    else:
        return input.unsqueeze(0)


def preprocess(input: Union[Tensor, Dict[str, Tensor]], space: spaces.Space) -> Union[Tensor, Dict[str, Tensor]]:
    """
    Preprocess to be to a neural network.

    For images, it normalizes the values by dividing them by 255 (to have values in [0, 1]), and transpose if needed as
    PyTorch use channel first format.
    For discrete observations, it create a one hot vector.

    :param input: Input with shape (N, ...)
    :param space: Space
    :return: The preprocessed tensor.
    """
    if isinstance(space, spaces.Box):
        if is_image_space(space):
            if space.shape[2] in [1, 3]:  # channel last -> make channel first
                input = torch.permute(input, (0, 3, 1, 2))
            return input.float() / 255.0
        else:
            return input.float().flatten(1)  # (N, ...) -> (N, D)

    elif isinstance(space, spaces.Discrete):
        # One hot encoding and convert to float to avoid errors
        return F.one_hot(input.long(), num_classes=space.n).float()

    elif isinstance(space, spaces.MultiDiscrete):
        # Tensor concatenation of one hot encodings of each Categorical sub-space
        return torch.cat(
            [
                F.one_hot(obs_.long(), num_classes=int(space.nvec[idx])).float()
                for idx, obs_ in enumerate(torch.split(input.long(), 1, dim=-1))
            ],
            dim=-1,
        ).view(*input.shape[:-1], sum(space.nvec))

    elif isinstance(space, spaces.MultiBinary):
        return input.float().flatten(1)  # (N, ...) -> (N, D)

    elif isinstance(space, spaces.Dict):
        # Do not modify by reference the original observation
        preprocessed = {}
        for key, _obs in input.items():
            preprocessed[key] = preprocess(_obs, space[key])
        return preprocessed

    else:
        raise NotImplementedError(f"Preprocessing not implemented for {space}")
