import numpy as np
import torch as th
import torch.nn.functional as F
from go_explore.rnd.rnd import Network, PredictorLearner
from gym.spaces import Box
from stable_baselines3.common.buffers import ReplayBuffer


def test_train_network():
    predictor = Network(3, 16, 3)
    target = Network(3, 16, 3)
    buffer = ReplayBuffer(10, Box(-1, 1, (3,)), Box(-1, 1, (2,)))
    cb = PredictorLearner(predictor, target, buffer, train_freq=10, grad_step=10, weight_decay=1e-5, lr=1e-3, batch_size=5)
    obs = np.array(
        [
            [1.02, -0.39, -1.47],
            [1.61, -1.33, -0.13],
            [-1.54, -1.47, -0.26],
            [-1.18, 0.53, 3.18],
            [-0.13, -0.37, -0.46],
            [-0.34, 0.13, -0.71],
            [-3.41, -0.72, 0.54],
            [0.7, -2.04, 0.57],
            [-1.13, 0.44, -0.71],
            [0.56, -0.21, -1.13],
        ]
    )
    actions = np.array(
        [
            [0.06, 0.45],
            [-0.82, -2.09],
            [0.97, -0.4],
            [-1.03, -0.5],
            [0.37, 0.37],
            [1.04, -0.62],
            [-1.38, 1.44],
            [-0.47, 0.38],
            [-0.76, -0.33],
            [0.65, -0.83],
        ]
    )
    for i in range(9):
        buffer.add(obs[i], obs[i + 1], actions[i], 0.0, False, [{}])

    pred = predictor(th.tensor(obs[0]).float())
    targ = target(th.tensor(obs[0]).float())
    loss_before = F.mse_loss(pred, targ)
    cb.train_once()
    pred = predictor(th.tensor(obs[0]).float())
    targ = target(th.tensor(obs[0]).float())
    loss_after = F.mse_loss(pred, targ)
    assert loss_before > loss_after
