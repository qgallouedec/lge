import numpy as np
from go_explore.go_explore.archive import ArchiveBuffer
from stable_baselines3.common.callbacks import BaseCallback


class LogNbCellsCallback(BaseCallback):
    """
    Callback used to log the number of cells encountered.

    :param archive: The buffer from which the number of cells is counted
    :param verbose: Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, archive: ArchiveBuffer, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.archive = archive
        self.subgoal_reached = []
        self.goal_reached = []

    def _on_step(self) -> bool:
        self.logger.record("cells/number", int(self.archive.nb_cells))
        return super()._on_step()


class SaveNbCellsCallback(BaseCallback):
    """
    Save number of cells every t timesteps into self.nb_cells.

    :param buffer: The buffer from which the number of cells is counted
    :param save_fq: The frequency of saving
    :param verbose: Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, buffer: ArchiveBuffer, save_fq: int = 1, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.buffer = buffer
        self.save_fq = save_fq
        self.nb_cells = []

    def _on_step(self) -> bool:
        if self.model.num_timesteps % self.save_fq == 0:
            self.nb_cells.append(self.buffer.nb_cells)
        return super()._on_step()
