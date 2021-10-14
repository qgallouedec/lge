import gym.spaces
import numpy as np

from go_explore.buffer import PathfinderBuffer
from go_explore.cell_computers import CellIsObs

OBS_SPACE = gym.spaces.Dict(
    achieved_goal=gym.spaces.Box(-1, 1, (3,)),
    desired_goal=gym.spaces.Box(-1, 1, (3,)),
    observation=gym.spaces.Box(-1, 1, (3,)),
)
ACTION_SPACE = gym.spaces.Box(-1, 1, (2,), np.float32)


def test_pathfinder_add():
    """
    Test Pathfinder add
    """
    buffer = PathfinderBuffer(100, OBS_SPACE, ACTION_SPACE, CellIsObs())
    obs = dict(
        achieved_goal=np.array([0.0, 0.0, 0.0]),
        desired_goal=np.array([0.0, 0.0, 0.0]),
        observation=np.array([0.0, 0.0, 0.0]),
    )
    next_obs = dict(
        achieved_goal=np.array([0.0, 0.0, 0.0]),
        desired_goal=np.array([0.0, 0.0, 0.0]),
        observation=np.array([0.0, 0.0, 0.0]),
    )
    action = np.array([0.0, 0.0])
    reward = 0.0
    done = False
    info = {}
    buffer.add(obs, next_obs, action, reward, done, [info])


# @pytest.mark.parametrize("bar", [bar1, bar2])
# def test_foo(bar):
