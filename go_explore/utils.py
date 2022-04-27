from typing import List, Optional

import numpy as np
import torch
from PIL import Image


def indexes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Indexes of a in b.

    :param a: Array of shape (...)
    :param b: Array of shape (N x ...)
    :return: Indexes of the occurences of a in b
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

    :param a: Array of shape (...)
    :param b: Array of shape (N x ...)
    :return: index of the first occurence of a in b
    """
    idxs = indexes(a, b)
    if idxs.shape[0] == 0:
        return None
    else:
        return idxs[0]


def multinomial(weights: torch.Tensor) -> torch.Tensor:
    p = weights / weights.sum()
    idx = torch.multinomial(p, 1)[0]
    return idx


def sample_geometric(mean: int, max_value: int) -> int:
    """
    Geometric sampling with some modifications.

    (1) The sampled value is < max_value.
    (2) The mean cannot be below max_value/20.
        If it is the case, the mean is replaced by max_value/20.

    :param mean: Mean of the geometric distribution
    :param max_value: Maximum value for the sample
    :return: Sampled value
    """
    # Clip the mean by 1/20th of the max value
    mean = np.clip(mean, a_min=int(max_value / 20), a_max=None)
    while True:  # loop until a correct value is found
        # for a geometric distributon, p = 1/mean
        value = np.random.geometric(1 / mean)
        if value < max_value:
            return value


def build_image(images: List[torch.Tensor]) -> Image:
    """
    Stack and return an image.

    :param images: List of batch of images. Each element must have size N x 3 x H x W
    :return: Image.
    """
    # Clamp the values to [0, 1]
    images = [torch.clamp(image, min=0.0, max=1.0) for image in images]

    # Tensor to array, and transpose
    images = [np.moveaxis(image.detach().cpu().numpy(), 1, 3) for image in images]

    # Stack all images
    rows = [np.hstack(tuple(image)) for image in images]
    full_image = np.vstack(rows)

    # Convert to Image
    full_image = Image.fromarray((full_image.squeeze() * 255).astype(np.uint8), "RGB")
    return full_image


def one_hot(x: np.ndarray, num_classes: int = -1) -> np.ndarray:
    """
    Numpy implementation of one_hot.

    :param x: class values of any shape.
    :param num_classes: Total number of classes. If set to -1, the number
        of classes will be inferred as one greater than the largest class
        value in the input array.
    :return: Array that has one more dimension with 1 values at the
    index of last dimension indicated by the input, and 0 everywhere
    else.

    Examples:
        >>> one_hot(np.arange(0, 5) % 3)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.],
               [1., 0., 0.],
               [0., 1., 0.]])
    """
    num_classes = np.max(x) + 1 if num_classes == -1 else num_classes
    y = np.eye(num_classes)[x]
    return y
