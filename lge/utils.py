import warnings
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
    Sample from a geometric distribution, but with the maximum value capped at `max_value`.

    Args:
        p (float): The probability of success of an individual trial.
        max_value (int): Maximum value for the sample, included.
        size (Optional[Union[int, Tuple[int]]]): Output size. If None, sample a single int.

    Returns:
        Union[int, np.ndarray]: Sampled value.

    Example:
        >>> sample_geometric_with_max(0.5, 5)
        2
        >>> sample_geometric_with_max(0.5, 5, size=(3,))
        array([2, 3, 2])
    """
    if p > 0:
        for _ in range(10_000):
            sample = np.random.geometric(p, size)
            if np.all(sample <= max_value):
                return sample
    return np.random.randint(0, max_value + 1, size)


def estimate_density(x: Tensor, samples: Tensor) -> Tensor:
    """
    Estimate the log density of `x` within the dataset.

    The estimated density is computed using the formula:

        density = k / (n * Cd) * dist_to_kst ** (-d)

    where:
        - k: The number of nearest neighbor to consider
        - n: The number of samples in the dataset
        - Cd: A constant that depends on the dimension of the input data.
              It is equal to  pi ** (d / 2) / exp(gamma(d / 2 + 1))
        - dist_to_kst: The distance of each point in `x` to its kth nearest neighbor in `samples`

    As this function is only used to order observations by their density,
    we save computational time by just returning -dist_to_kst
    (as this function is monotonically increasing with respect to density)

    Args:
        x (Tensor): Points to evaluate density
        samples (Tensor): The samples from the distribution to estimate

    Returns:
        Tensor:  The estimate density on `x`, up to a monotonic transformation.
    """
    # In the original paper, the recommended value is k = int(2 * n ** (1 / d)). However, it is common to have
    # multiple identical values in the dataset. Consequently, most points would have a k-st closest dist to zero.
    # That's why we take a larger k. Empiraically, k = 1000 seem to work well.
    k = min(1000, len(samples))
    cdist = torch.cdist(x, samples)
    dist_to_kst = cdist.topk(k, largest=False)[0][:, -1]
    return -dist_to_kst


def lighten(arr: np.ndarray, threshold: float) -> np.ndarray:
    """
    Returns the indexes of the input array such that all successive elements are
    at least distant from the threshold value. It keeps the last one in place.

    Args:
        arr (np.ndarray): Input array.
        threshold (float): Distance threshold.

    Returns:
        np.ndarray: List of indexes.

    Examples:
        >>> arr = np.array([4.0, 5.0, 5.1, 6.0, 7.0])
        >>> idxs = lighten(arr, threshold=1.0)
        >>> idxs
        array([0, 1, 3, 4])
        >>> arr[idxs]
        array([4., 5., 6., 7.])
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

    Args:
        space (spaces.Space): Space

    Raises:
        ValueError: If space is not in [(Multi)Discrete, MultiBinary, Box]

    Returns:
        Tuple[int]: The shape
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

    Raises warning when trying to get the size of an image.

    Args:
        space (spaces.Space): Space

    Returns:
        int: Size
    """
    if is_image_space(space):
        warnings.warn("Why are you computing the size of an image?")
    return np.prod(get_shape(space))


def is_batched(input: Union[np.ndarray, int], space: spaces.Space) -> bool:
    """
    Whether the input is batched, meaning it is a batch of elements of the space.

    Args:
        input (np.ndarray): Observation or batch of values
        space (spaces.Space): Space from which the values come

    Raises:
        ValueError: When the observation(s) does/don't come from the space
        ValueError: When space is not in [(Multi)Discrete, MultiBinary, Box]

    Returns:
        bool: Whether the input is batched
    """
    if isinstance(space, (spaces.Box, spaces.MultiDiscrete, spaces.MultiBinary)):
        if input.shape == space.shape:
            return False
        elif input.shape[1:] == space.shape:
            return True
        else:
            raise ValueError(f"Wrong input shape")
    elif isinstance(space, spaces.Discrete):
        if isinstance(input, int) or len(input.shape) == 0:
            return False
        elif len(input.shape) == 1:
            return True
        else:
            raise ValueError(f"Wrong input shape")
    else:
        raise ValueError(f"Space {space} not supported.")


def batchify(input: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[np.ndarray, Dict[str, Tensor]]:
    """
    Make the input a batch.

    Args:
        input (Union[np.ndarray, Dict[str, np.ndarray]]): Unbatched input.

    Returns:
        Union[np.ndarray, Dict[str, np.ndarray]]: Batched version of the input.

    Examples:
        >>> batchify(torch.tensor([1, 2, 3]))
        tensor([[1, 2, 3]])
        >>> batchify({"key": torch.tensor([1, 2, 3])})
        {'key': tensor([[1, 2, 3]])}
    """
    if isinstance(input, dict):
        return {key: batchify(value) for key, value in input.items()}
    else:
        return np.expand_dims(input, 0)


def preprocess(input: Union[Tensor, Dict[str, Tensor]], space: spaces.Space) -> Union[Tensor, Dict[str, Tensor]]:
    """
    Preprocess the input before passing it through the neural network.

    Args:
        input (Union[Tensor, Dict[str, Tensor]]): Batched input
        space (spaces.Space): Space from which the values come

    Raises:
        NotImplementedError: If space is not in [(Multi)Discrete, MultiBinary, Box]

    Returns:
        Union[Tensor, Dict[str, Tensor]]: Preprocessed input
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


def maybe_make_channel_first(observation: np.ndarray) -> np.ndarray:
    """
    Make channel first when observation is an image.

    Args:
        observation (np.ndarray): Observation (batched)

    Returns:
        np.ndarray: If observation is as image, it makes it channel-first
    """
    if len(observation.shape) == 4:
        if observation.shape[3] in [1, 3]:
            return np.transpose(observation, (0, 3, 1, 2))
    return observation


def maybe_transpose(observations: np.ndarray, observation_space: spaces.Space) -> np.ndarray:
    """
    Transpoose image so that the observation fits the observation space.

    Args:
        observations (np.ndarray): Batched observations
        observation_space (spaces.Space): Space

    Returns:
        np.ndarray: Batched observation that fit the observation space shape
    """
    if is_image_space(observation_space):
        if observation_space.shape[0] in [1, 3]:
            # channel first
            if observation_space.shape[0] == observations.shape[3]:
                return np.transpose(observations, (0, 3, 1, 2))
        elif observation_space.shape[2] in [1, 3]:
            # channel flastirst
            if observation_space.shape[2] == observations.shape[1]:
                return np.transpose(observations, (0, 2, 3, 1))
    return observations
