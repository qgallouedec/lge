from stable_baselines3.common.type_aliases import ReplayBufferSamples
from go_explore.icm.icm import ICM
import torch
import gym


def test_inverse_model():
    from go_explore.icm.models import InverseModel

    im = InverseModel(feature_dim=6, action_dim=3, hidden_dim=16)
    obs_feature = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    next_obs_feature = torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    pred_action = im(obs_feature, next_obs_feature)
    assert pred_action.shape == torch.Size([3])


def test_forward_model():
    from go_explore.icm.models import ForwardModel

    fm = ForwardModel(feature_dim=6, action_dim=3, hidden_dim=16)
    action = torch.tensor([0.3, 0.4, 0.5])
    obs_feature = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    pred_next_obs_feature = fm(action, obs_feature)
    assert pred_next_obs_feature.shape == torch.Size([6])


def test_feature_extractor():
    from go_explore.icm.models import FeatureExtractor

    fe = FeatureExtractor(obs_dim=7, feature_dim=6, hidden_dim=16)
    obs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    obs_feature = fe(obs)
    assert obs_feature.shape == torch.Size([6])


def test_learn_forward_model():
    from go_explore.icm.models import ForwardModel

    fm = ForwardModel(feature_dim=6, action_dim=3, hidden_dim=16)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(fm.parameters())
    actions = torch.tensor(
        [
            [0.78, -1.62, -1.42],
            [-0.34, 0.06, -0.33],
            [1.17, 0.66, 0.32],
            [-1.81, 0.86, 0.18],
            [0.46, 0.41, 0.97],
            [0.33, -1.74, 0.8],
            [0.44, -1.04, 0.48],
            [-0.08, 0.32, -0.26],
            [-0.6, -1.89, -1.64],
            [0.71, 0.03, 2.28],
        ]
    )
    obs_features = torch.tensor(
        [
            [-0.14, 1.07, -0.85, -0.63, 0.99, 0.13],
            [0.02, -1.35, 1.13, 0.61, 0.26, 1.95],
            [1.04, 1.11, 0.54, -0.06, -0.68, 0.16],
            [-0.12, 1.12, 1.02, -1.75, -0.03, -0.42],
            [-1.65, 1.57, 0.31, -1.04, -0.54, -0.89],
            [1.31, 0.91, 0.5, -1.0, 0.48, 0.77],
            [-0.84, -0.63, -0.24, -0.62, 1.74, -0.01],
            [-0.44, -0.21, 2.0, -0.68, 1.36, 0.97],
            [-0.75, 0.29, -1.12, -0.1, 0.58, 1.36],
            [0.71, -0.54, 0.93, -0.2, -0.07, 1.1],
        ]
    )
    next_obs_features = torch.tensor(
        [
            [0.88, 2.07, 1.08, 0.76, -0.76, 0.19],
            [-0.71, -1.84, -0.01, 0.01, 0.27, -2.32],
            [-0.21, -1.52, 1.48, -0.01, -1.13, 1.05],
            [-1.0, 0.19, 1.03, -1.73, 0.41, -0.67],
            [0.07, -0.88, -0.12, 0.38, 0.01, 1.26],
            [-1.95, -0.67, 1.61, 0.18, 1.7, -0.67],
            [1.23, -1.09, -0.56, -1.74, 0.22, 0.39],
            [-1.97, -0.16, -0.33, -1.02, 0.14, 0.86],
            [0.14, 0.89, 0.66, -0.79, 1.09, 0.45],
            [0.13, -0.12, -0.9, 1.19, -0.44, 0.64],
        ]
    )

    for _ in range(1000):
        pred_next_obs_feature = fm(actions, obs_features)
        loss = criterion(next_obs_features, pred_next_obs_feature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss < 0.1:
            return
    raise Exception("Learning failed")


def test_learn_inverse_model():
    from go_explore.icm.models import InverseModel

    im = InverseModel(feature_dim=6, action_dim=3, hidden_dim=16)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(im.parameters())
    obs_features = torch.tensor(
        [
            [-0.14, 1.07, -0.85, -0.63, 0.99, 0.13],
            [0.02, -1.35, 1.13, 0.61, 0.26, 1.95],
            [1.04, 1.11, 0.54, -0.06, -0.68, 0.16],
            [-0.12, 1.12, 1.02, -1.75, -0.03, -0.42],
            [-1.65, 1.57, 0.31, -1.04, -0.54, -0.89],
            [1.31, 0.91, 0.5, -1.0, 0.48, 0.77],
            [-0.84, -0.63, -0.24, -0.62, 1.74, -0.01],
            [-0.44, -0.21, 2.0, -0.68, 1.36, 0.97],
            [-0.75, 0.29, -1.12, -0.1, 0.58, 1.36],
            [0.71, -0.54, 0.93, -0.2, -0.07, 1.1],
        ]
    )
    next_obs_features = torch.tensor(
        [
            [0.88, 2.07, 1.08, 0.76, -0.76, 0.19],
            [-0.71, -1.84, -0.01, 0.01, 0.27, -2.32],
            [-0.21, -1.52, 1.48, -0.01, -1.13, 1.05],
            [-1.0, 0.19, 1.03, -1.73, 0.41, -0.67],
            [0.07, -0.88, -0.12, 0.38, 0.01, 1.26],
            [-1.95, -0.67, 1.61, 0.18, 1.7, -0.67],
            [1.23, -1.09, -0.56, -1.74, 0.22, 0.39],
            [-1.97, -0.16, -0.33, -1.02, 0.14, 0.86],
            [0.14, 0.89, 0.66, -0.79, 1.09, 0.45],
            [0.13, -0.12, -0.9, 1.19, -0.44, 0.64],
        ]
    )
    actions = torch.tensor(
        [
            [0.78, -1.62, -1.42],
            [-0.34, 0.06, -0.33],
            [1.17, 0.66, 0.32],
            [-1.81, 0.86, 0.18],
            [0.46, 0.41, 0.97],
            [0.33, -1.74, 0.8],
            [0.44, -1.04, 0.48],
            [-0.08, 0.32, -0.26],
            [-0.6, -1.89, -1.64],
            [0.71, 0.03, 2.28],
        ]
    )

    for _ in range(1000):
        pred_action = im(obs_features, next_obs_features)
        loss = criterion(actions, pred_action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss < 0.1:
            return
    raise Exception("Learning failed")


def test_learn_feature_extractor():
    from go_explore.icm.models import FeatureExtractor

    fe = FeatureExtractor(obs_dim=7, feature_dim=6, hidden_dim=16)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(fe.parameters())
    obs = torch.tensor(
        [
            [0.17, -1.47, 0.99, 0.85, -0.81, -0.82, 0.56],
            [-1.39, -0.0, 0.33, -1.79, 0.84, -0.47, -1.02],
            [0.61, 0.57, -2.05, 0.7, -0.86, -0.71, -0.76],
            [0.43, -0.51, 1.39, 0.33, -1.82, -1.14, 0.87],
            [0.16, 0.46, -1.94, -0.15, -1.46, -0.66, -0.72],
            [-0.12, -1.35, -0.09, -0.19, 0.97, 1.91, -0.74],
            [-0.49, -0.13, 0.7, -0.09, 1.37, 0.86, 0.35],
            [1.0, -0.84, 1.54, -1.27, -0.68, 0.67, 1.27],
            [-1.0, 0.07, 0.93, -0.72, -1.71, -1.06, -0.06],
            [0.45, -0.6, 0.76, 0.19, -0.36, 0.38, -1.54],
        ]
    )
    obs_features = torch.tensor(
        [
            [-0.14, 1.07, -0.85, -0.63, 0.99, 0.13],
            [0.02, -1.35, 1.13, 0.61, 0.26, 1.95],
            [1.04, 1.11, 0.54, -0.06, -0.68, 0.16],
            [-0.12, 1.12, 1.02, -1.75, -0.03, -0.42],
            [-1.65, 1.57, 0.31, -1.04, -0.54, -0.89],
            [1.31, 0.91, 0.5, -1.0, 0.48, 0.77],
            [-0.84, -0.63, -0.24, -0.62, 1.74, -0.01],
            [-0.44, -0.21, 2.0, -0.68, 1.36, 0.97],
            [-0.75, 0.29, -1.12, -0.1, 0.58, 1.36],
            [0.71, -0.54, 0.93, -0.2, -0.07, 1.1],
        ]
    )

    for _ in range(1000):
        pred_obs_features = fe(obs)
        loss = criterion(obs_features, pred_obs_features)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss < 0.1:
            return
    raise Exception("Learning failed")


def test_learn_icm():
    icm = ICM(
        scaling_factor=1.0,
        actor_loss_coef=1.0,
        inverse_loss_coef=0.5,
        forward_loss_coef=0.5,
        obs_dim=7,
        action_dim=3,
        feature_dim=6,
        hidden_dim=16,
    )
    optimizer = torch.optim.Adam(icm.parameters())
    observations = torch.tensor(
        [
            [0.17, -1.47, 0.99, 0.85, -0.81, -0.82, 0.56],
            [-1.39, -0.0, 0.33, -1.79, 0.84, -0.47, -1.02],
            [0.61, 0.57, -2.05, 0.7, -0.86, -0.71, -0.76],
            [0.43, -0.51, 1.39, 0.33, -1.82, -1.14, 0.87],
            [0.16, 0.46, -1.94, -0.15, -1.46, -0.66, -0.72],
            [-0.12, -1.35, -0.09, -0.19, 0.97, 1.91, -0.74],
            [-0.49, -0.13, 0.7, -0.09, 1.37, 0.86, 0.35],
            [1.0, -0.84, 1.54, -1.27, -0.68, 0.67, 1.27],
            [-1.0, 0.07, 0.93, -0.72, -1.71, -1.06, -0.06],
            [0.45, -0.6, 0.76, 0.19, -0.36, 0.38, -1.54],
        ]
    )
    actions = torch.tensor(
        [
            [0.78, -1.62, -1.42],
            [-0.34, 0.06, -0.33],
            [1.17, 0.66, 0.32],
            [-1.81, 0.86, 0.18],
            [0.46, 0.41, 0.97],
            [0.33, -1.74, 0.8],
            [0.44, -1.04, 0.48],
            [-0.08, 0.32, -0.26],
            [-0.6, -1.89, -1.64],
            [0.71, 0.03, 2.28],
        ]
    )
    next_observations = torch.tensor(
        [
            [0.79, -1.04, -0.25, -0.16, 0.36, 0.16, 0.0],
            [-0.33, 0.36, -0.21, -0.18, 0.38, 0.56, -0.79],
            [1.17, -1.84, -1.06, 1.97, -0.22, 1.19, 1.02],
            [-1.59, -0.23, 0.74, 0.43, 0.1, -1.4, 0.31],
            [-0.94, -1.36, 0.28, 0.36, -1.95, 0.18, -1.21],
            [-0.09, -0.47, -0.39, 0.7, 0.86, 2.53, -1.57],
            [-1.44, 0.38, 1.04, -0.79, -0.27, 1.6, 0.88],
            [-0.59, -0.07, -0.06, -0.74, 0.62, -1.44, -0.8],
            [-0.47, -0.6, 0.04, -0.19, -0.56, 0.54, 1.49],
            [-0.64, 1.28, -1.65, 1.15, 0.01, 0.4, -1.03],
        ]
    )
    dones = torch.tensor([False, False, False, False, False, False, False, False, False, False])
    rewards = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    replay_data = ReplayBufferSamples(observations, actions, next_observations, dones, rewards)
    mean_modified_reward = icm.modify_reward(replay_data).rewards.mean()
    if not mean_modified_reward - rewards.mean() > 0.03:  # if the intrinsic reward is below 0.03, it is really weird
        raise Exception("Modified reward should be geater than reward before training.")
    for _ in range(1000):
        loss = icm.modify_loss(torch.tensor(1.0), replay_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if not abs(loss - 1.0) < 0.01:
        # loss should converge toward beta * constant_actor_loss
        raise Exception("Learning failed")
    mean_modified_reward = icm.modify_reward(replay_data).rewards.mean()
    if not abs(mean_modified_reward - rewards.mean()) < 0.001:
        raise Exception("Intrinsic reward should be gone after training ICM")


def test_ICM():
    import panda_gym
    from go_explore.common.wrappers import UnGoalWrapper
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = DummyVecEnv([lambda: UnGoalWrapper(gym.make("PandaReach-v2"))])
    env = VecNormalize(env, norm_reward=False)

    icm = ICM(
        scaling_factor=1.0,
        actor_loss_coef=1.0,
        inverse_loss_coef=0.5,
        forward_loss_coef=0.5,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        feature_dim=6,
        hidden_dim=16,
    )

    model = SAC("MlpPolicy", env, actor_loss_modifier=icm)
    model.learn(1000, eval_env=env, eval_freq=1000, n_eval_episodes=10, reward_modifier=icm)


test_learn_icm()
