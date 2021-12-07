import gym.spaces
import numpy as np

from go_explore.go_explore.archive import ArchiveBuffer
from go_explore.go_explore.cell_computers import Cell, CellIsObs

OBS_SPACE = gym.spaces.Box(-1, 1, (2,), np.float32)
ACTION_SPACE = gym.spaces.Box(-1, 1, (2,), np.float32)


def test_archive_add():
    cell_computer = CellIsObs()
    archive = ArchiveBuffer(100, OBS_SPACE, ACTION_SPACE, cell_computer)
    first = True
    obs = np.array([0.0, 0.0])
    next_obs = np.array([1.0, 1.0])
    action = np.array([0.0, 0.0])
    reward = 0.0
    done = False
    info = {}
    archive.add(obs, next_obs, action, reward, done, [info], first)

    current_cell = cell_computer.compute_cell(np.array([0.0, 0.0]))
    next_cell = cell_computer.compute_cell(np.array([1.0, 1.0]))
    assert archive._cell_to_idx[current_cell] == 0
    assert archive._cell_to_idx[next_cell] == 1
    assert len(archive._cell_to_obss[current_cell]) == 1
    assert (archive._cell_to_obss[current_cell][0] == np.array([0.0, 0.0])).all()
    assert (archive._counts == np.array([1, 1])).all()
    assert archive._idx_to_cell[0] == current_cell
    assert archive._idx_to_cell[1] == next_cell
    assert (archive.csgraph == np.array([[np.inf, 1.0], [np.inf, np.inf]])).all()


def test_archive_reset1():
    cell_computer = CellIsObs()
    archive = ArchiveBuffer(100, OBS_SPACE, ACTION_SPACE, cell_computer)

    for _ in range(3):
        first = True
        obs = np.array([0.0, 0.0])
        for i in range(1, 4):
            next_obs = np.array([float(i), float(i)])
            action = np.array([0.0, 0.0])
            reward = 0.0
            done = False
            info = {}
            archive.add(obs, next_obs, action, reward, done, [info], first)
            first = False
            obs = next_obs

    cell0 = cell_computer.compute_cell(np.array([0.0, 0.0]))
    cell1 = cell_computer.compute_cell(np.array([1.0, 1.0]))
    cell2 = cell_computer.compute_cell(np.array([2.0, 2.0]))
    cell3 = cell_computer.compute_cell(np.array([3.0, 3.0]))
    assert archive._cell_to_idx[cell0] == 0
    assert archive._cell_to_idx[cell1] == 1
    assert archive._cell_to_idx[cell2] == 2
    assert archive._cell_to_idx[cell3] == 3
    assert len(archive._cell_to_obss[cell0]) == 3
    assert len(archive._cell_to_obss[cell1]) == 3
    assert len(archive._cell_to_obss[cell2]) == 3
    assert len(archive._cell_to_obss[cell3]) == 3
    assert (archive._cell_to_obss[cell0][0] == np.array([0.0, 0.0])).all()
    assert (archive._cell_to_obss[cell0][1] == np.array([0.0, 0.0])).all()
    assert (archive._cell_to_obss[cell1][0] == np.array([1.0, 1.0])).all()
    assert (archive._cell_to_obss[cell1][1] == np.array([1.0, 1.0])).all()
    assert (archive._cell_to_obss[cell2][0] == np.array([2.0, 2.0])).all()
    assert (archive._cell_to_obss[cell2][1] == np.array([2.0, 2.0])).all()
    assert (archive._cell_to_obss[cell3][0] == np.array([3.0, 3.0])).all()
    assert (archive._cell_to_obss[cell3][1] == np.array([3.0, 3.0])).all()
    assert (archive._counts == np.array([3.0, 3.0, 3.0, 3.0])).all()
    assert archive._idx_to_cell[0] == cell0
    assert archive._idx_to_cell[1] == cell1
    assert archive._idx_to_cell[2] == cell2
    assert archive._idx_to_cell[3] == cell3
    csgraph = np.array(
        [
            [np.inf, 1.0, np.inf, np.inf],
            [np.inf, np.inf, 1.0, np.inf],
            [np.inf, np.inf, np.inf, 1.0],
            [np.inf, np.inf, np.inf, np.inf],
        ]
    )
    assert (archive.csgraph == csgraph).all()


def test_archive_reset2():
    cell_computer = CellIsObs()
    archive = ArchiveBuffer(100, OBS_SPACE, ACTION_SPACE, cell_computer)
    trajectories = [
        [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.array([1.0, 2.0])],
        [np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0])],
    ]
    for trajectory in trajectories:
        first = True
        for i in range(len(trajectory) - 1):
            obs = trajectory[i]
            next_obs = trajectory[i + 1]
            action = np.array([0.0, 0.0])
            reward = 0.0
            done = False
            info = {}
            archive.add(obs, next_obs, action, reward, done, [info], first)
            first = False

    cell0 = cell_computer.compute_cell(np.array([0.0, 0.0]))
    cell1 = cell_computer.compute_cell(np.array([1.0, 0.0]))
    cell2 = cell_computer.compute_cell(np.array([1.0, 1.0]))
    cell3 = cell_computer.compute_cell(np.array([1.0, 2.0]))
    cell4 = cell_computer.compute_cell(np.array([0.0, 1.0]))
    assert archive._cell_to_idx[cell0] == 0
    assert archive._cell_to_idx[cell1] == 1
    assert archive._cell_to_idx[cell2] == 2
    assert archive._cell_to_idx[cell3] == 3
    assert archive._cell_to_idx[cell4] == 4
    assert len(archive._cell_to_obss[cell0]) == 2
    assert len(archive._cell_to_obss[cell1]) == 2
    assert len(archive._cell_to_obss[cell2]) == 1
    assert len(archive._cell_to_obss[cell3]) == 1
    assert len(archive._cell_to_obss[cell4]) == 1
    assert (archive._cell_to_obss[cell0][0] == np.array([0.0, 0.0])).all()
    assert (archive._cell_to_obss[cell0][1] == np.array([0.0, 0.0])).all()
    assert (archive._cell_to_obss[cell1][0] == np.array([1.0, 0.0])).all()
    assert (archive._cell_to_obss[cell1][1] == np.array([1.0, 0.0])).all()
    assert (archive._cell_to_obss[cell2][0] == np.array([1.0, 1.0])).all()
    assert (archive._cell_to_obss[cell3][0] == np.array([1.0, 2.0])).all()
    assert (archive._cell_to_obss[cell4][0] == np.array([0.0, 1.0])).all()
    assert (archive._counts == np.array([2.0, 2.0, 1.0, 1.0, 1.0])).all()
    assert archive._idx_to_cell[0] == cell0
    assert archive._idx_to_cell[1] == cell1
    assert archive._idx_to_cell[2] == cell2
    assert archive._idx_to_cell[3] == cell3
    assert archive._idx_to_cell[4] == cell4
    csgraph = np.array(
        [  # [0, 0] [1, 0] [1, 1] [1, 2] [0, 1]
            [np.inf, 1.0, np.inf, np.inf, 1.0],
            [1.0, np.inf, 1.0, np.inf, np.inf],
            [np.inf, np.inf, np.inf, 1.0, np.inf],
            [np.inf, np.inf, np.inf, np.inf, np.inf],
            [np.inf, np.inf, np.inf, np.inf, np.inf],
        ]
    )
    assert (archive.csgraph == csgraph).all()


def test_path_sampling():
    cell_computer = CellIsObs()
    archive = ArchiveBuffer(100, OBS_SPACE, ACTION_SPACE, cell_computer)
    trajectories = [
        [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.array([1.0, 2.0])],
        [np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0])],
    ]
    for trajectory in trajectories:
        first = True
        for i in range(len(trajectory) - 1):
            obs = trajectory[i]
            next_obs = trajectory[i + 1]
            action = np.array([0.0, 0.0])
            reward = 0.0
            done = False
            info = {}
            archive.add(obs, next_obs, action, reward, done, [info], first)
            first = False

    path = archive.sample_subgoal_path(np.array([0.0, 0.0]))
    path = [list(obs) for obs in path]  # convinient to compare
    possible_pathes = [
        [[1.0, 0.0]],
        [[1.0, 0.0], [1.0, 1.0]],
        [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]],
        [[0.0, 1.0]],
    ]
    assert path in possible_pathes

    # ensure that all possible pathes are sampled
    sampled_pathes = []
    for _ in range(200):  # 200 sampled path seems enough to be sure that all possible pathes are sampled
        path = archive.sample_subgoal_path(np.array([0.0, 0.0]))
        path = [list(obs) for obs in path]  # convinient to compare
        sampled_pathes.append(path)
    for path in possible_pathes:
        assert path in sampled_pathes
