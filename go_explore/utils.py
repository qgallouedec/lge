from typing import Iterable, List, Optional, Union

import cv2
import gym
import numpy as np
import torch
from gym import Env, spaces
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback

ATARI_ACTIONS = [
    "NOOP",
    "FIRE",
    "UP",
    "RIGHT",
    "LEFT",
    "DOWN",
    "UPRIGHT",
    "UPLEFT",
    "DOWNRIGHT",
    "DOWNLEFT",
    "UPFIRE",
    "RIGHTFIRE",
    "LEFTFIRE",
    "DOWNFIRE",
    "UPRIGHTFIRE",
    "UPLEFTFIRE",
    "DOWNRIGHTFIRE",
    "DOWNLEFTFIRE",
]


def indexes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Indexes of a in b.

    :param a: Array of shape (...)
    :param b: Array of shape (N x ...)
    :return: Indexes of the occurences of a in b
    """
    if b.shape[0] == 0:
        return np.array([])
    a = a.flatten()
    b = b.reshape((b.shape[0], -1))
    idxs = np.where((a == b).all(1))[0]
    return idxs


def index(a: np.ndarray, b: np.ndarray) -> Optional[int]:
    """
    Index of first occurence of a in b.

    :param a: Array of shape (...)
    :param b: Array of shape (N x ...)
    :return: index of the first occurence of a in b
    """
    idxs = indexes(a, b)
    if idxs.shape[0] == 0:
        return None
    else:
        return idxs[0]


def multinomial(weights: torch.Tensor) -> torch.Tensor:
    p = weights / weights.sum()
    idx = torch.multinomial(p, 1)[0]
    return idx


def sample_geometric(mean: int, max_value: int) -> int:
    """
    Geometric sampling with some modifications.

    (1) The sampled value is < max_value.
    (2) The mean cannot be below max_value/20.
        If it is the case, the mean is replaced by max_value/20.

    :param mean: Mean of the geometric distribution
    :param max_value: Maximum value for the sample
    :return: Sampled value
    """
    # Clip the mean by 1/20th of the max value
    mean = np.clip(mean, a_min=int(max_value / 20), a_max=None)
    while True:  # loop until a correct value is found
        # for a geometric distributon, p = 1/mean
        value = np.random.geometric(1 / mean)
        if value < max_value:
            return value


def build_image(images: List[torch.Tensor]) -> Image:
    """
    Stack and return an image.

    :param images: List of batch of images. Each element must have size N x 3 x H x W
    :return: Image.
    """
    # Clamp the values to [0, 1]
    images = [torch.clamp(image, min=0.0, max=1.0) for image in images]

    # Tensor to array, and transpose
    images = [np.moveaxis(image.detach().cpu().numpy(), 1, 3) for image in images]

    # Stack all images
    rows = [np.hstack(tuple(image)) for image in images]
    full_image = np.vstack(rows)

    # Convert to Image
    full_image = Image.fromarray((full_image.squeeze() * 255).astype(np.uint8), "RGB")
    return full_image


def is_image(x: torch.Tensor) -> bool:
    """Whether the input is an image, or a batch of images"""
    shape = x.shape
    if len(shape) >= 3 and 3 in shape:
        return True
    else:
        return False


def human_format(num: int) -> str:
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format("{:f}".format(num).rstrip("0").rstrip("."), ["", "k", "M", "B", "T"][magnitude])


class ImageSaver(BaseCallback):
    def __init__(self, env: Env, save_freq: int) -> None:
        super(ImageSaver, self).__init__()
        self.env = env
        self.save_freq = save_freq

    def _on_step(self) -> None:
        if self.n_calls % self.save_freq == 0:
            img = Image.fromarray(self.env.render("rgb_array"))
            img.save(human_format(self.n_calls) + ".bmp")
        return super()._on_step()


class AtariWrapper(gym.Wrapper):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.

    :param env: the environment
    :param width:
    :param height:
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 12), dtype=env.observation_space.dtype
        )

    def step(self, action: np.ndarray) -> np.ndarray:
        """

        :param frame: environment frame
        :return: the observation
        """
        tot_reward = 0
        obs_buf = np.zeros((self.height, self.width, 12), dtype=self.observation_space.dtype)

        for frame_idx in range(4):
            frame, reward, done, info = super().step(action)
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            obs_buf[:, :, 3 * frame_idx : 3 * (frame_idx + 1)] = frame
            tot_reward += reward
        return obs_buf, tot_reward, done, info

    def reset(self):
        obs_buf = np.zeros((self.height, self.width, 12), dtype=self.observation_space.dtype)
        frame = super().reset()
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        obs_buf[:, :, 0:3] = frame
        for frame_idx in range(1, 4):
            frame, _, _, _ = super().step(0)
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            obs_buf[:, :, 3 * frame_idx : 3 * (frame_idx + 1)] = frame

        return obs_buf


def detect_private_eye_end_frame(obs: np.ndarray) -> bool:
    """Return whether the agent is at the frame on the border of the ma in PrivateEye."""
    hmin, hmax = 65, 72
    lmin, lmax = 54, 60

    # If and only if the agent is in this room in the game, this square shoould be equal to that values:
    ref = np.array(
        [
            [[114, 126, 45], [135, 163, 62], [135, 169, 69], [135, 183, 84], [135, 183, 84], [135, 183, 84]],
            [[149, 149, 43], [140, 140, 32], [140, 140, 32], [140, 140, 32], [140, 147, 40], [140, 179, 76]],
            [[170, 170, 53], [191, 191, 55], [191, 191, 55], [191, 191, 55], [191, 191, 55], [191, 191, 55]],
            [[134, 134, 43], [207, 207, 62], [204, 204, 61], [204, 204, 61], [204, 204, 61], [204, 204, 61]],
            [[158, 158, 40], [137, 137, 30], [134, 134, 29], [134, 134, 29], [134, 134, 29], [134, 134, 29]],
            [[134, 134, 29], [134, 134, 29], [134, 134, 29], [134, 134, 29], [134, 134, 29], [134, 134, 29]],
            [[134, 134, 29], [134, 134, 29], [134, 134, 29], [134, 134, 29], [134, 134, 29], [134, 134, 29]],
        ],
        dtype=np.uint8,
    )
    return (obs[hmin:hmax, lmin:lmax] == ref).all()
