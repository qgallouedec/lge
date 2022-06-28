import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from torch import optim

from lge.inverse_model import ConvInverseModel


def sample_image():
    # Sample fake Atari image (stacking of 4 frames)
    img = np.random.randint(0, 255, (4, 4, 3)).astype(np.float32)
    img = cv2.resize(img, (84, 84))  # (4 x 4) to (84 x 84)
    img = np.moveaxis(img, 2, 0)  # (H x W x C) to (C x H x W)
    return img


def test_train_inverse_dynamic():
    action_space = spaces.Discrete(10)

    n_obs = 3  # Number of possible observations
    all_observations = torch.tensor([sample_image() for _ in range(n_obs)]) / 255

    # For each possible couple (observation, next_observation), we define an action
    all_actions = torch.tensor([[action_space.sample() for _ in range(n_obs)] for _ in range(n_obs)])

    inverse_model = ConvInverseModel(action_size=action_space.n, latent_size=4)
    optimizer = optim.Adam(inverse_model.parameters(), lr=1e-3)

    for _ in range(100):
        batch_size = 32
        # Sample
        obs_idx = torch.randint(0, 3, size=(batch_size,))
        next_obs_idx = torch.randint(0, 3, size=(batch_size,))
        observations = all_observations[obs_idx]
        next_observations = all_observations[next_obs_idx]
        actions = all_actions[obs_idx, next_obs_idx]

        # Compute the output image
        inverse_model.train()
        pred_actions = inverse_model(observations, next_observations)

        # Compute the loss
        loss = F.cross_entropy(pred_actions, actions.squeeze())

        # Step the optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss < 0.75
