import gym
import numpy as np

from go_explore.go_explore.archive import ArchiveBuffer


def test_import():
    import go_explore.envs


def test_continuous_minigrid1():
    import go_explore.envs

    env = gym.make("ContinuousMinigrid-v0")
    env.reset()
    action = env.action_space.sample()
    env.step(action)


def test_continuous_minigrid2():
    import go_explore.envs

    env = gym.make("ContinuousMinigrid-v0")
    obs = env.reset()
    action = [1.0, 1.0]
    next_obs, reward, done, info = env.step(action)
    assert (obs == np.array([0.0, 0.0])).all()
    assert (next_obs == np.array([1.0, 1.0])).all()
    action = [0.8, -0.3]
    next_obs, reward, done, info = env.step(action)
    assert (next_obs == np.array([2.0, 1.0])).all()
    action = [-1.5, -0.8]
    next_obs, reward, done, info = env.step(action)
    assert (next_obs == np.array([1.0, 0.0])).all()
    obs = env.reset()
    assert (obs == np.array([0.0, 0.0])).all()


def test_subgoal_continuous_minigrid1():
    import go_explore.envs

    env = gym.make("SubgoalContinuousMinigrid-v0")
    obs = env.reset()
    assert (obs["observation"] == np.array([0.0, 0.0])).all()
    for _ in range(20):
        desired_goal = obs["desired_goal"]
        action = obs["desired_goal"] - obs["observation"]
        obs, reward, done, info = env.step(action)
        if (desired_goal == obs["observation"]).all():
            break
    assert reward == 0.0
    assert info.get("is_success", False)
    assert not done  # 1 goal remains
    for _ in range(20):
        desired_goal = obs["desired_goal"]
        action = obs["desired_goal"] - obs["observation"]
        obs, reward, done, info = env.step(action)
        if (desired_goal == obs["observation"]).all():
            break
    assert reward == 0.0
    assert info.get("is_success", False)
    assert done  # last goal reached


def test_panda_subgoal_random():
    import go_explore.envs
    from go_explore.go_explore.cell_computers import PandaCellComputer

    env = gym.make("PandaSubgoalRandom-v0")
    cell_computer = PandaCellComputer()

    obs = env.reset()
    for _ in range(50):
        desired_cell = cell_computer.compute_cell(obs["desired_goal"])
        current_pos = obs["observation"][:3]
        desired_pos = obs["desired_goal"][:3]
        action = np.zeros(4)
        action[:3] = 3 * (desired_pos - current_pos)
        obs, reward, done, info = env.step(action)
        current_cell = cell_computer.compute_cell(obs["observation"])
        if current_cell == desired_cell:
            break
    assert reward == 0.0
    assert info.get("is_success", False)
    assert not done  # 1 goal remains
    for _ in range(50):
        desired_cell = cell_computer.compute_cell(obs["desired_goal"])
        current_pos = obs["observation"][:3]
        desired_pos = obs["desired_goal"][:3]
        action = np.zeros(4)
        action[:3] = 3 * (desired_pos - current_pos)
        obs, reward, done, info = env.step(action)
        current_cell = cell_computer.compute_cell(obs["observation"])
        if current_cell == desired_cell:
            break
    assert reward == 0.0
    assert info.get("is_success", False)
    assert done  # last goal reached


def test_panda_subgoal_archive():
    import go_explore.envs
    from go_explore.go_explore.cell_computers import PandaCellComputer

    env = gym.make("PandaSubgoalArchive-v0")
    archive: ArchiveBuffer = env.archive
    cell_computer = PandaCellComputer()

    fake_episode1 = np.array(
        [
            [0.04, -0.0, 0.2, 0.0, -0.0, 0.0, 0.0],
            [-0.06, 0.07, 0.2, -0.43, 0.53, 0.25, 0.0],
            [-0.04, 0.11, 0.19, 0.47, -0.52, 0.02, 0.03],
            [-0.03, 0.13, 0.17, 0.7, 0.41, 0.27, 0.07],
            [-0.06, 0.07, 0.11, -1.1, 0.09, -0.26, 0.0],
            [-0.08, 0.04, 0.01, 0.31, -0.34, -0.13, 0.0],
            [-0.15, 0.04, 0.01, -0.38, -0.15, 0.1, 0.0],
            [-0.16, 0.03, 0.01, -1.78, 0.19, 0.21, 0.01],
            [-0.15, 0.11, 0.0, -0.33, 0.55, 0.03, 0.08],
            [-0.15, 0.13, 0.0, 0.19, -0.55, -0.02, 0.02],
            [-0.16, 0.18, 0.07, 0.84, -0.89, 1.03, 0.08],
            [-0.18, 0.17, 0.01, -0.48, -0.0, -0.49, 0.02],
            [-0.22, 0.17, 0.02, -1.3, -0.98, -0.78, 0.02],
        ]
    )
    fake_episode2 = np.array(
        [
            [0.04, -0.0, 0.2, 0.0, -0.0, 0.0, 0.0],
            [0.03, -0.04, 0.13, 0.06, -0.12, 0.34, 0.01],
            [0.05, -0.08, 0.01, -0.05, -0.29, 0.02, 0.07],
            [0.1, -0.05, 0.01, -0.02, 0.18, -0.32, 0.08],
            [0.11, -0.06, 0.07, -0.18, -0.13, 0.16, 0.0],
            [0.05, -0.05, 0.01, -0.48, 0.03, -0.1, 0.01],
            [-0.02, -0.08, 0.06, -0.09, -0.13, 0.82, 0.01],
            [-0.12, -0.22, 0.03, -0.61, -0.68, -0.17, 0.04],
            [-0.24, -0.21, 0.03, -0.63, 0.18, -0.93, 0.04],
            [-0.27, -0.18, 0.02, 0.13, -0.3, 0.11, 0.0],
            [-0.31, -0.1, 0.02, -0.38, 0.29, -0.27, 0.07],
            [-0.34, -0.03, 0.02, 0.71, 0.3, 0.52, 0.0],
            [-0.28, -0.1, 0.02, -0.15, -0.22, -0.43, 0.07],
        ]
    )
    for fake_episode in [fake_episode1, fake_episode2]:
        episode_start = True
        for i in range(len(fake_episode) - 1):
            archive.add(fake_episode[i], fake_episode[i + 1], np.zeros(4), 0.0, False, [{"episode_start": episode_start}])
            episode_start = False

    obs = env.reset()
    for _ in range(50):
        desired_cell = cell_computer.compute_cell(obs["desired_goal"])
        current_pos = obs["observation"][:3]
        desired_pos = obs["desired_goal"][:3]
        action = np.zeros(4)
        action[:3] = 3 * (desired_pos - current_pos)
        obs, reward, done, info = env.step(action)
        current_cell = cell_computer.compute_cell(obs["observation"])
        if current_cell == desired_cell:
            assert reward == 0.0
            assert info.get("is_success", False)
        if done:
            break
    assert done  # last goal reached


def test_done_delay():
    import go_explore.envs
    from go_explore.go_explore.cell_computers import PandaCellComputer

    env = gym.make("PandaSubgoalRandom-v0", done_delay=3)
    cell_computer = PandaCellComputer()

    obs = env.reset()
    for _ in range(50):
        desired_cell = cell_computer.compute_cell(obs["desired_goal"])
        current_pos = obs["observation"][:3]
        desired_pos = obs["desired_goal"][:3]
        action = np.zeros(4)
        action[:3] = 3 * (desired_pos - current_pos)
        obs, reward, done, info = env.step(action)
        current_cell = cell_computer.compute_cell(obs["observation"])
        if current_cell == desired_cell:
            break
    assert reward == 0.0
    assert info.get("is_success", False)
    assert not done  # 1 goal remains
    for _ in range(50):
        desired_cell = cell_computer.compute_cell(obs["desired_goal"])
        current_pos = obs["observation"][:3]
        desired_pos = obs["desired_goal"][:3]
        action = np.zeros(4)
        action[:3] = 3 * (desired_pos - current_pos)
        obs, reward, done, info = env.step(action)
        current_cell = cell_computer.compute_cell(obs["observation"])
        if current_cell == desired_cell:
            break
    assert reward == 0.0
    assert info.get("is_success", False)
    assert not done  # last goal reached but done countdown
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    assert not done  # not yet
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    assert not done  # still not yet
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    assert done  # now!


test_done_delay()
