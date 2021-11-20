from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d


def save_dat(
    values: np.ndarray,
    timesteps: np.ndarray = None,
    step: int = 1,
    filename: str = "result.dat",
    target_length: Optional[int] = None,
) -> None:
    """
    Process and save a numpy array into a .dat file.

    :param values: 2D array. First axis for experiments, second axis for timesteps
    :type values: np.ndarray
    :param step: Number of timesteps between values, defaults to 1.
    :type step: int, optional
    :param filename: Filename for saving, defaults to "result.dat".
    :type filename: str, optional
    :param target_length: The target number of elements in the output file. Downsample if necessary.
        If None, stores every inputs elements.
    :type target_length: int or None, optional
    """
    timesteps = timesteps if timesteps is not None else np.arange(values.shape[1]) * step
    if target_length is not None:
        old_idx = np.arange(values.shape[1])
        f = interp1d(old_idx, old_idx, kind="nearest")
        new_idx = np.arange(target_length) * values.shape[1] / target_length
        new_idx = f(new_idx).astype(np.int64)
        timesteps = timesteps[new_idx]
        values = values[:, new_idx]
    med = np.median(values, axis=0)
    lowq = np.quantile(values, 0.33, axis=0)
    highq = np.quantile(values, 0.66, axis=0)
    out = np.vstack((timesteps, med, lowq, highq)).transpose()

    np.savetxt(filename, out, fmt="%d %.3f %.3f %.3f", header="timestep med lowq highq", comments="")


def load_eval(file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads evaluations.npz and returns timestepds and results

    :param file: File path.
    :type file: str
    :return: Timesteps and results.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    timesteps = np.load(file)["timesteps"]
    results = np.load(file)["results"]
    results = results.mean(1) # mean over all the eval episodes
    return timesteps, results


def load_evals(files: List[str]):
    all_timesteps_results = [load_eval(file) for file in files]
    timesteps = all_timesteps_results[0][0]
    results = [timesteps_results[1] for timesteps_results in all_timesteps_results]
    print([len(a) for a in results])
    results = np.stack(results)
    print(len(timesteps), len(results))
    return timesteps, results


if __name__ == "__main__":
    timesteps, results = load_evals(["./results/" + str(i) + "/evaluations.npz" for i in range(7)])
    save_dat(results, timesteps)
