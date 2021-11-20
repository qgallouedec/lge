from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
from numpy.core.fromnumeric import squeeze


class Cell:
    """
    Cells is used as dict key, thus it must be hashable.
    """

    def __init__(self, arr: np.ndarray) -> None:
        self.arr = arr.astype(np.float64) + np.zeros_like(arr)  # avoid -0.
        self._hash = hash(self.arr.tobytes())

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Cell):
            return False
        return hash(self) == hash(other)

    def __neq__(self, other: Any) -> bool:
        return not self == other

    def __repr__(self) -> str:
        return "Cell({})".format(str(self.arr))


class CellComputer(ABC):
    """
    Abstract class used to compute the cell of an observation.

    To define a new cell computer, you only need to inherit this class with your `_process` method.
    """

    def compute_cell(self, observation: np.ndarray) -> Cell:
        """
        Compute the cell associated with one observation.

        :param observation: The observation to convert as cell
        :return: The obtained cell.
        """
        observations = observation.reshape((1, -1))
        cell = self.compute_cells(observations)[0]
        return cell

    def compute_cells(self, observations: np.ndarray) -> List[Cell]:
        """
        Compute the cells associated with the observations.

        :param observations: Observations as a array. Each row is an observation
        :return: The obtained list of cells.
        """
        processed_obs = self._process(observations)
        cells = [Cell(val) for val in processed_obs]
        return cells

    @abstractmethod
    def _process(self, observations: np.ndarray) -> np.ndarray:
        """
        Process the observation.

        :param observations: The observations to be processed. Each row is one observation
        :return: The processed observations.
        """
        raise NotImplementedError()


class DownsampleCellComputer(CellComputer):
    """
    Reduce and downsample the observation: floor(coef*x/std).

    :param std: The observation standard deviation
    :param coef: The multiplication coeficient applied before computing floor. The higher, the more cells
    """

    def __init__(self, std: np.ndarray, coef: float) -> None:
        super().__init__()
        self.std = std
        self.coef = coef

    def _process(self, observations: np.ndarray) -> np.ndarray:
        downsampled_obs = np.floor(self.coef * observations / self.std)
        return downsampled_obs


class CellIsObs(CellComputer):
    """
    The observation is taken as cell. Not recommended for continuous observation space.
    """

    def _process(self, observations: np.ndarray) -> np.ndarray:
        return observations


class PandaCellComputer(CellComputer):
    """
    Cell computer for PandaNoTask-v1. A cell is computed based on the ee position and the gripper opening.
    """

    def __init__(self) -> None:
        self.std = 0.20 * np.ones(3)
        super().__init__()

    def _process(self, observations: np.ndarray) -> np.ndarray:
        reduced_obs = observations[..., [3, 4, 5]]  # x, y, z
        downsampled_obs = np.floor(reduced_obs / self.std, dtype=np.float32)
        return downsampled_obs


class PandaObjectCellComputer(CellComputer):
    """
    Cell computer for PandaNoTask-v1. A cell is computed based on the ee position and the gripper opening.
    """

    def __init__(self) -> None:
        self.std = 0.20 * np.ones(6)
        super().__init__()

    def _process(self, observations: np.ndarray) -> np.ndarray:
        reduced_obs = observations[..., [3, 4, 5, 10, 11, 12]]  # x, y, z object
        downsampled_obs = np.floor(reduced_obs / self.std, dtype=np.float32)
        return downsampled_obs


class MountainCarCellComputer(CellComputer):
    """
    Cell computer for MountainCarContinuous-v0. A cell is computed based on the ee position.
    """

    def __init__(self) -> None:
        self.std = np.array([1.0])
        self.coef = 10.0
        super().__init__()

    def _process(self, observations: np.ndarray) -> np.ndarray:
        reduced_obs = observations[..., [0]]  # take only the position
        downsampled_obs = np.floor(self.coef * reduced_obs / self.std)
        return downsampled_obs
