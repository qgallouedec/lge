from typing import Callable, Tuple

import numpy as np
import torch as th
from torchvision.transforms.functional import resize, rgb_to_grayscale

MAX_W = 160
MAX_H = 210
MAX_NB_SHADES = 255


def distribution_score(probs: th.Tensor, nb_images: int, split_factor: int) -> float:
    """
    The objective function for candidate downscaling parameters.

    O(p, n) = H_n(p) / L(n, T), where
    - H_n(p) is the entropy ratio with the uniform distribution: -sum_i p_i*log(p_i)/log(n)
    - L(n, T) is the discrepancy measure between the desired number T of cells and the number of
        cells obtained n: √(|n/T - 1| + 1)

    :param probs: The probabilities of the distribution. Sum must be 1.
    :param nb_images: The number of images of the sample that produced this distribution
    :param split_factor: The desired ratio between the number of images of the sample and the
        number of produced cells.
    """
    if len(probs) == 1:
        return 0.0
    target_nb_cells = split_factor * nb_images
    nb_cells = probs.shape[0]
    entropy_ratio = -th.sum(probs * th.log(probs) / np.log(nb_cells))
    discrepancy_measure = np.sqrt(np.abs(nb_cells / target_nb_cells - 1) + 1)
    return entropy_ratio / discrepancy_measure


def sample_geometric(mean: int, max_value: int) -> int:
    """
    Geometric sampling with some modifications.

    (1) The sampled value cannot exceed max_value.
    (2) The mean cannot be below max_value/20.
        If it is the case, the mean is replaced by max_value/20.

    :param mean: The mean of the geometric distribution
    :param max_value: Maximum value for the sample
    :return: Sampled value
    """
    # Clip the mean by 1/20th of the max value
    mean = np.clip(mean, a_min=int(max_value / 20), a_max=None)
    while True:  # loop until a correct value is found
        # for a geometric distributon, p = 1/mean
        value = np.random.geometric(1 / mean)
        if value > 0 and value < max_value:
            return value


def get_cells(images: th.Tensor, width: int, height: int, nb_shades: int) -> th.Tensor:
    """
    Return the cells associated with each image.

    :param images: The images as a Tensor of dims (N x H x W x C)
    :param width: The width of the downscaled image
    :param height: The height of the downscaled image
    :param nb_shades: Number of possible shades of gray in the cell representation
    :return: The cells as a Tensor
    """
    # Converts (N x H x W x C) to (N x C x H x W)
    images = images.transpose(-3, -1)
    # Range [0, 255] to range [0.0, 1.0]
    images = images / 255
    # Convert to grayscale
    images = rgb_to_grayscale(images)
    # Resize
    prev_shape = images.shape[:-2]
    images = images.reshape((-1, *images.shape[-2:])) #  (... x 1 x H x W) to (N x H x W)
    images = resize(images, (width, height))
    images = images.reshape((*prev_shape, *images.shape[-2:])) #  (... x 1 x H x W) to (N x H x W)
    # Downscale
    cells = th.floor(images * nb_shades)
    # Squeeze (... x 1 x H x W) to (... x H x W)
    cells = cells.squeeze(-3)
    return cells


def get_param_score(images: th.Tensor, width: int, height: int, nb_shades: int) -> float:
    """
    Get the score of the parameters.

    :param images: The images as a Tensor of dims (N x H x W x C)
    :param width: The width of the downscaled image
    :param height: The height of the downscaled image
    :param nb_shades: Number of possible shades of gray in the cell representation
    :return: The score
    """
    cells = get_cells(images, width, height, nb_shades)
    # List the uniques cells produced, and compute their probability
    cells, counts = th.unique(cells, return_counts=True, dim=0)
    # Get the probability distribution produced
    nb_images = images.shape[0]
    probs = counts / nb_images
    # Compute the score
    score = distribution_score(probs, nb_images, split_factor=0.125)
    return score


def optimize_downscale_parameters(
    images: np.ndarray,
    best_w: int = MAX_W,
    best_h: int = MAX_H,
    best_nb_shades: int = MAX_NB_SHADES,
    nb_trials: int = 3000,
) -> Tuple[int, int, int]:
    """
    Find the best parameters for the cell computation.

    :param images: Frames, dimensions are (N, W, H, C)
    :param best_w: Best known width value, defaults to MAX_W
    :param best_h: Best known height value, defaults to MAX_H
    :param best_nb_shades: Best known number of shades value, defaults to MAX_NB_SHADES
    :param nb_trials: Number of trials to find best parameters, defaults to 3000
    :return: New best width, height, and number of shades
    """
    # Compute the current best score
    best_score = get_param_score(images, best_w, best_h, best_nb_shades)
    # Try new parameters
    param_tried = set()
    while len(param_tried) < nb_trials:
        # Sample
        width = sample_geometric(best_w, MAX_W)
        height = sample_geometric(best_h, MAX_H)
        nb_shades = sample_geometric(best_nb_shades, MAX_NB_SHADES)

        # If the params has already been tried, skip and sample new set of params
        if (width, height, nb_shades) in param_tried:
            continue
        else:
            param_tried.add((width, height, nb_shades))

        # Get the score of the parameters, and update the best if necessary
        score = get_param_score(images, width, height, nb_shades)
        if score > best_score:
            best_score = score
            best_w = width
            best_h = height
            best_nb_shades = nb_shades

    return best_w, best_h, best_nb_shades


CellFactory = Callable[[th.Tensor], th.Tensor]


class DownscaleCellFactory:
    def __init__(self) -> None:
        self.width = MAX_W
        self.height = MAX_H
        self.nb_shades = MAX_NB_SHADES

    def set_parameters(self, width: int, height: int, nb_shades: int) -> None:
        """
        Set the parameters of the cell factory.

        :param width: _description_
        :param height: _description_
        :param nb_shades: _description_
        """
        self.width = width
        self.height = height
        self.nb_shades = nb_shades

    def __call__(self, images: th.Tensor) -> th.Tensor:
        if len(images.shape) == 3:  # (H x W x C)
            images = images.unsqueeze(0)  # (H x W x C) to (1 x H x W x C)
            cells = get_cells(images, self.width, self.height, self.nb_shades)
            return cells[0]
        elif len(images.shape) > 3:
            return get_cells(images, self.width, self.height, self.nb_shades)
        else:
            raise ValueError("Dimensions should be (H x W x C) or (N x H x W x C)")
