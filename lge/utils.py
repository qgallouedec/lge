from typing import List, Optional

import numpy as np
import torch
from gym import Env
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback
from torch import Tensor


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


def sample_geometric_with_max(p, max_value, size=None):
    """
    Sample follow geometric law, but are below the max_value.

    :param p: The probability of success of an individual trial
    :param max_value: Maximum value for the sample, included
    :param size: Output shape
    :return: Sampled value
    """
    if p > 0:
        for _ in range(10_000):
            sample = np.random.geometric(p, size)
            if np.all(sample <= max_value):
                return sample
    return np.random.randint(0, max_value + 1)


def build_image(images: List[Tensor]) -> Image.Image:
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


def is_image(x: Tensor) -> bool:
    """Whether the input is an image, or a batch of images"""
    shape = x.shape
    if len(shape) >= 3 and 3 in shape:
        return True
    else:
        return False


def human_format(num: int) -> str:
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format("{:f}".format(num).rstrip("0").rstrip("."), ["", "k", "M", "B", "T"][magnitude])


class ImageSaver(BaseCallback):
    def __init__(self, env: Env, save_freq: int) -> None:
        super(ImageSaver, self).__init__()
        self.env = env
        self.save_freq = save_freq

    def _on_step(self) -> None:
        if self.n_calls % self.save_freq == 0:
            img = Image.fromarray(self.env.render("rgb_array"))
            img.save(human_format(self.n_calls) + ".bmp")
        return super()._on_step()


def round(input: Tensor, decimals: float) -> Tensor:
    """
    Rounding, but extended to every float.

    :param input: Input tensor
    :param decimals: Decimals, can be float
    :return: The rounded tensor
    :rtype: Tensor

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
    :rtype: Tensor
    """
    n, d = samples.shape
    k = int(2 * n ** (1 / d))
    cdist = torch.cdist(x, samples)
    dist_to_kst = cdist.topk(k, largest=False)[0][:, -1]
    return dist_to_kst ** (-d)


def lighten(arr, threshold):
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
