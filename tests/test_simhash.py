import numpy as np
from go_explore.simhash.simhash import SimHash, SimHashMotivation


def test_simhash():
    simhash = SimHash(obs_size=4, granularity=16)
    # close obs should have the same hash
    obs1 = np.array([0.1, 0.2, 0.3, 0.4])
    obs2 = np.array([0.101, 0.2, 0.3, 0.4])
    assert (simhash(obs1) == simhash(obs2)).all()
    # distant obs should not have the same hash
    obs1 = np.array([0.1, 0.2, 0.3, 0.4])
    obs2 = np.array([0.4, 0.3, 0.2, 0.1])
    assert (simhash(obs1) != simhash(obs2)).any()


def test_simhash_motivation():
    reward_modifier = SimHashMotivation(obs_dim=4, granularity=16, beta=1)
    new_reward = reward_modifier.modify_reward(
        obs=np.array([[1.0, 2.0, 3.0, 4.0]]),
        action=np.array([1.0]),
        next_obs=np.array([[1.0, 2.0, 3.0, 5.0]]),
        reward=1.0,
    )
    assert new_reward == 1.0 + 1.0  # since the next_obs has never been encountered
    new_reward = reward_modifier.modify_reward(
        obs=np.array([[1.0, 2.0, 3.0, 4.0]]),
        action=np.array([1.0]),
        next_obs=np.array([[1.01, 2.0, 3.0, 5.0]]),
        reward=1.0,
    )

    assert new_reward == 1.0 + 1.0 / np.sqrt(2.0)  # since a close next_obs has already been encountered
