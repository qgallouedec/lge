import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Type

import gym.spaces
import numpy as np
import optuna
import torch as th
from gym import spaces
from torchvision.transforms.functional import resize, rgb_to_grayscale
from go_explore.utils import sample_geometric

optuna.logging.set_verbosity(optuna.logging.WARNING)


def distribution_score(probs: th.Tensor, nb_samples: int, split_factor: float) -> float:
    """
    Get the score of the distribution. Used to find the best cell factory parameters.

    O(p, n) = H_n(p) / L(n, T), where
    - H_n(p) is the entropy ratio with the uniform distribution: -sum_i p_i*log(p_i)/log(n)
    - L(n, T) is the discrepancy measure between the desired number T of cells and the number of
        cells obtained n: √(|n/T - 1| + 1)

    :param probs: The probabilities of the distribution. Sum must be 1.
    :param nb_samples: The number of samples that produced this distribution
    :param split_factor: The desired ratio between the number of samples and the number of produced cells.
    """
    if len(probs) == 1:
        return 0.0
    target_nb_cells = split_factor * nb_samples
    nb_cells = probs.shape[0]
    entropy_ratio = -th.sum(probs * th.log(probs) / np.log(nb_cells)).item()
    discrepancy_measure = np.sqrt(np.abs(nb_cells / target_nb_cells - 1) + 1)
    return entropy_ratio / discrepancy_measure


class CellFactory(ABC):
    cell_space: gym.spaces.Space

    @abstractmethod
    def __call__(self, observations: th.Tensor) -> th.Tensor:
        ...

    @abstractmethod
    def optimize_param(self, samples: th.Tensor, nb_trials: int = 300) -> float:
        ...


def get_param_score(cells: th.Tensor) -> float:
    """
    Get the score of the parameters.

    :param cells: The cells obtained from the samples.
    :return: The score
    """
    # List the uniques cells produced, and compute their probability
    _, counts = th.unique(cells, return_counts=True, dim=0)
    nb_samples = cells.shape[0]
    probs = counts / nb_samples
    # Compute the score
    score = distribution_score(probs, nb_samples, split_factor=0.125)
    return score


class ImageGrayscaleDownscale(CellFactory):
    """
    The image is scaled to gray, resized smaller, and the number of shades is lowered.

    :param height: The height of the downscaled image
    :param width: The width of the downscaled image
    :param nb_shades: Number of possible shades of gray in the cell representation

    Example:
    >>> cell_factory = ImageGrayscaleDownscale(height=15, width=10,  nb_shades=20)
    >>> images.shape
    torch.Size([10, 3, 210, 160])  # (N x 3 x H x W)
    >>> cell_factory(images).shape
    torch.Size([10, 15, 10])  # (N x NEW_H x NEW_W)
    """

    MAX_H = 210
    MAX_W = 160
    MAX_NB_SHADES = 255

    def __init__(self, height: int = MAX_H, width: int = MAX_W, nb_shades: int = MAX_NB_SHADES) -> None:
        self.set_param(height, width, nb_shades)

    def set_param(self, height: int, width: int, nb_shades: int) -> None:
        self.height = height
        self.width = width
        self.nb_shades = nb_shades
        self.cell_space = spaces.Box(low=0, high=255, shape=(height * width,))

    def __call__(self, images: th.Tensor) -> th.Tensor:
        """
        Return the cells associated with each image.

        :param images: The images as a Tensor of dims (... x 3 x H x W)
        :param height: The height of the downscaled image
        :param width: The width of the downscaled image
        :param nb_shades: Number of possible shades of gray in the cell representation
        :return: The cells as a Tensor
        """
        # Image's dims must be (... x 3 x H x W)
        if images.shape[-3] != 3 and images.shape[-1] == 3:  # (... x H x W x 3)
            images = images.moveaxis(-1, -3)  # (... x H x W x 3) to # (... x 3 x H x W)
            # Yes, it does not work with W == 3 but, come on...
        # We need a little trick on shape, because resize  and rgb_to_grayscale only accepts size (N x H x W)
        prev_shape = images.shape[:-3]  # the "..." part of the shape
        images = images.reshape((-1, *images.shape[-3:]))  #  (... x 3 x H x W) to (N x 3 x H x W)
        # Convert to grayscale
        images = rgb_to_grayscale(images)  # (N x 1 x H x W)
        # Resize
        images = resize(images, (self.height, self.width))  # (N x 1 x NEW_W x NEW_H)
        images = images.reshape((*prev_shape, -1))  #  (N x 1 x H x W) to (... x H x W)
        # Downscale
        coef = 256 / self.nb_shades
        cells = (th.floor(images / coef) * coef).to(th.uint8)
        return cells

    def optimize_param(self, samples: th.Tensor, nb_trials: int = 300) -> float:
        def objective(trial: optuna.Trial):
            height = trial.suggest_int("height", 1, self.MAX_H)
            width = trial.suggest_int("width", 1, self.MAX_W)
            nb_shades = trial.suggest_int("nb_shades", 1, self.MAX_NB_SHADES)
            self.set_param(height, width, nb_shades)
            cells = self.__call__(samples)
            score = get_param_score(cells)
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=nb_trials)
        self.set_param(**study.best_params)
        return study.best_value

    def old_optimize_param(self, samples: th.Tensor, nb_trials: int = 300) -> float:
        """
        Find the best parameters for the cell computation.

        :param images: The images as a Tensor of dims (N x 3 x H x W)
        :param best_h: Best known height value, defaults to MAX_H
        :param best_w: Best known width value, defaults to MAX_W
        :param best_nb_shades: Best known number of shades value, defaults to MAX_NB_SHADES
        :param nb_trials: Number of trials to find best parameters, defaults to 3000
        :return: New best height, width, and number of shades
        """
        best_h, best_w, best_nb_shades, best_score = self.MAX_H, self.MAX_W, self.MAX_NB_SHADES, 0
        # Try new parameters
        param_tried = set()
        while len(param_tried) < nb_trials:
            # Sample
            height = sample_geometric(best_h, self.MAX_H)
            width = sample_geometric(best_w, self.MAX_W)
            nb_shades = sample_geometric(best_nb_shades, self.MAX_NB_SHADES)

            # If the params has already been tried, skip and sample new set of params
            if (height, width, nb_shades) in param_tried:
                continue
            else:
                param_tried.add((height, width, nb_shades))

            # Get the score of the parameters, and update the best if necessary
            self.set_param(height, width, nb_shades)
            cells = self.__call__(samples)
            score = get_param_score(cells)

            if score > best_score:
                best_score = score
                best_h = height
                best_w = width
                best_nb_shades = nb_shades
        self.set_param(best_h, best_w, best_nb_shades)
        return best_score


class CellIsObs(CellFactory):
    """
    Cell is observation.

    Example:
    >>> observation_space.shape
    (3, 4)
    >>> cell_factory = CellIsObs(observation_space)
    >>> (cell_factory(observations) == observation).all()
    True
    """

    def __init__(self, observation_space: spaces.Space) -> None:
        self.cell_space = copy.deepcopy(observation_space)

    def __call__(self, observations: th.Tensor) -> th.Tensor:
        """
        Compute the cells.

        :param observations: Observations
        :return: A tensor of cells
        """
        return observations.clone()

    def optimize_param(cls, samples: th.Tensor, nb_trials: int = 300) -> Dict:
        return dict()


class DownscaleObs(CellFactory):
    """
    Downscale the observation.

    Example:
    >>> observation_space.shape
    (3, 4)
    >>> cell_factory = DownscaleObs(observation_space)

    """

    def __init__(self, observation_space: spaces.Space) -> None:
        self.cell_space = copy.deepcopy(observation_space)
        self.steps = np.ones(self.cell_space.shape[0])

    def __call__(self, observations: th.Tensor) -> th.Tensor:
        """
        Compute the cells.

        :param observations: Observations
        :return: A tensor of cells
        """
        cells = th.floor(observations / self.steps) * self.steps
        return cells

    def optimize_param(self, samples: th.Tensor, nb_trials: int = 300) -> float:
        def objective(trial: optuna.Trial):
            steps = []
            for dim in range(len(self.steps)):
                steps.append(trial.suggest_loguniform("step_" + str(dim), 1e-6, 1e4))
            self.steps = np.array(steps)
            cells = self.__call__(samples)
            score = get_param_score(cells)
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=nb_trials)
        self.steps = np.array([study.best_params["step_" + str(dim)] for dim in range(len(self.steps))])
        return study.best_value
