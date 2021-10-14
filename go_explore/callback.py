from abc import abstractmethod

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.callbacks import BaseCallback

from go_explore.buffer import PathfinderBuffer


class StoreCallback(BaseCallback):
    """
    A callback used to store the transitions in a buffer.

    :param buffer: The buffer in which the transitions are stored
    :param verbose: Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, buffer: BaseBuffer, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.buffer = buffer

    def _on_step(self) -> bool:
        obs = self.locals["self"]._last_obs
        next_obs = self.locals["new_obs"]
        action = self.locals["action"]
        reward = self.locals["reward"]
        done = self.locals["done"]
        infos = self.locals["infos"]
        if done and infos[0].get("terminal_observation") is not None:
            next_obs = infos[0]["terminal_observation"]
        self.buffer.add(obs, next_obs, action, reward, done, infos)
        return super()._on_step()


class EpisodeEndCallback(BaseCallback):
    """
    Base class for callbacks called at the end of episodes.

    Define _on_episode_end() to define the desired behaviour.

    :param verbose: Verbosity level 0: not output 1: info 2: debug
    """

    def _on_step(self) -> bool:
        if self.locals["done"][0]:
            return self._on_episode_end()
        else:
            return True

    @abstractmethod
    def _on_episode_end(self) -> None:
        """
        This method will be called by the model after the end of a episode.

        :return: If the callback returns False, training is aborted early.
        """
        raise NotImplementedError()


class StopTrainingOnMaxTimesteps(BaseCallback):
    """
    Stop the training once a maximum number of timesteps are played.

    :param max_timesteps: Maximum number of timesteps to stop training
    :param verbose: Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, max_timesteps: int, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.max_timesteps = max_timesteps

    def _on_step(self) -> bool:
        num_timesteps = self.n_calls
        continue_training = num_timesteps < self.max_timesteps
        return continue_training


class LogNbCellsCallback(BaseCallback):
    """
    Callback used to log the number of cells encountered.

    :param buffer: The buffer from which the number of cells is counted
    :param verbose: Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, buffer: PathfinderBuffer, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.buffer = buffer
        self.nb_cells = []

    def _on_step(self) -> bool:
        self.logger.record("cells/number", int(self.buffer.nb_cells))
        if self.model.num_timesteps % 500 == 0:
            self.nb_cells.append(self.buffer.nb_cells)
        return super()._on_step()


class SaveNbCellsCallback(BaseCallback):
    """
    Save number of cells every t timesteps into self.nb_cells.

    :param buffer: The buffer from which the number of cells is counted
    :param save_fq: The frequency of saving
    :param verbose: Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, buffer: PathfinderBuffer, save_fq: int = 1, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.buffer = buffer
        self.save_fq = save_fq
        self.nb_cells = []

    def _on_step(self) -> bool:
        if self.model.num_timesteps % self.save_fq == 0:
            self.nb_cells.append(self.buffer.nb_cells)
        return super()._on_step()


class StopTrainingOnEndTrajectory(BaseCallback):
    """
    Stop the training once a maximum number of timesteps are played.

    :param verbose: Verbosity level 0: not output 1: info 2: debug
    """

    def _on_step(self) -> bool:
        is_end_traj = self.locals["infos"][0].get("traj_success", False)
        return not is_end_traj
