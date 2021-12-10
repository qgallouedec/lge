import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
import torch
from go_explore.simhash.simhash import SimHash, SimHashMotivation
from gym import spaces


def test_simhash():
    simhash = SimHash(obs_size=4, granularity=16)
    # close obs should have the same hash
    obs1 = torch.tensor([0.1, 0.2, 0.3, 0.4])
    obs2 = torch.tensor([0.101, 0.2, 0.3, 0.4])
    assert (simhash(obs1) == simhash(obs2)).all()
    # distant obs should not have the same hash
    obs1 = torch.tensor([0.1, 0.2, 0.3, 0.4])
    obs2 = torch.tensor([0.4, 0.3, 0.2, 0.1])
    assert (simhash(obs1) != simhash(obs2)).any()


def test_simhash_motivation():
    buffer = ReplayBuffer(100, spaces.Box(-10, 10, (2,)), spaces.Box(-10, 10, (2,)))
    reward_modifier = SimHashMotivation(buffer, None, granularity=4, beta=1)
    observations = [np.array([[0.17, -1.47]]), np.array([[-1.39, -0.0]]), np.array([[0.61, 0.57]]), np.array([[-1.39, -0.0]])]
    actions = [np.array([[-0.46, -0.49]]), np.array([[1.33, -0.33]]), np.array([[1.33, -0.33]])]

    for i in range(len(observations) - 1):
        buffer.add(observations[i], observations[i + 1], actions[i], np.array([0.0]), np.array([False]), [{}])

    replay_data = ReplayBufferSamples(
        observations=torch.tensor([[0.17, -1.47], [-1.39, -0.0], [-1.39, -0.0]]),
        actions=torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        next_observations=torch.tensor([[-1.39, -0.0], [0.61, 0.57], [0.61, 0.57]]),
        dones=torch.tensor([False, False, False]),
        rewards=torch.tensor([0.0, 0.0, 0.0]),
    )
    new_replay_data = reward_modifier.modify_reward(replay_data)
    assert (new_replay_data.rewards == torch.ones(3) / torch.sqrt(torch.tensor([2.0, 1.0, 1.0]))).all()
