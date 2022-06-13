from typing import Dict, List, Optional, Union

import cv2
import gym
import numpy as np
import torch
from gym import Env, spaces
from gym.envs.atari import AtariEnv
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback

ATARI_ACTIONS = [
    "NOOP",  # 0
    "FIRE",  # 1
    "UP",  # 2
    "RIGHT",  # 3
    "LEFT",  # 4
    "DOWN",  # 5
    "UPRIGHT",  # 6
    "UPLEFT",  # 7
    "DOWNRIGHT",  # 8
    "DOWNLEFT",  # 9
    "UPFIRE",  # 10
    "RIGHTFIRE",  # 11
    "LEFTFIRE",  # 12
    "DOWNFIRE",  # 13
    "UPRIGHTFIRE",  # 14
    "UPLEFTFIRE",  # 15
    "DOWNRIGHTFIRE",  # 16
    "DOWNLEFTFIRE",  # 17
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


def sample_geometric_with_max(p, max_value, size=None):
    """
    Sample follow geometric law, but are below the max_value.

    :param p: The probability of success of an individual trial
    :param max_value: Maximum value for the sample
    :param size: Output shape
    :return: Sampled value
    """
    for _ in range(10_000):
        sample = np.random.geometric(p, size)
        if np.all(sample <= max_value):
            return sample
    raise ValueError(
        "Fail to sample geometric given p = {:.4f} and max_value = {:d} after 10_000 trials. Try to changes these values.".format(p, max_value)
    )


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
    if len(shape) >= 3 and 3 in shape or 12 in shape:
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


def get_montezuma_revenge_info(env: AtariEnv) -> Dict:
    """
    Get some infos from the RAM of the game.

    :param env: The MontezumaRevenge env
    :type env: AtariEnv
    :return: _description_
    :rtype: _type_
    """
    assert env.spec.id == "ALE/MontezumaRevenge-v5", "function only functional with Montezuma Revenge env."
    ram = env.ale.getRAM()
    item_str = format(ram[65], "8b")
    return dict(
        frame_number=ram[0],
        room_number=ram[3],
        x=ram[42],
        y=ram[43],
        score=10000 * ram[19] + 100 * ram[20] + ram[21],
        life_number=ram[58],
        have_amulet=item_str[7] == "1",
        have_key1=item_str[6] == "1",
        have_key2=item_str[5] == "1",
        have_key3=item_str[4] == "1",
        have_key4=item_str[3] == "1",
        have_sword=item_str[2] == "1",
        have_spare=item_str[1] == "1",
        have_torch=item_str[0] == "1",
    )


class MontezumaRevengeWrapper(AtariWrapper):
    def step(self, action):
        obs, reward, done, info = super().step(action)
        extra_info = get_montezuma_revenge_info(self.env)
        info = {**info, **extra_info}
        return obs, reward, done, info


def get_private_eye_info(env: AtariEnv) -> Dict:
    """
    Get some infos from the RAM of the game.

    :param env: The MontezumaRevenge env
    :type env: AtariEnv
    :return: _description_
    :rtype: _type_
    """
    assert env.spec.id == "ALE/PrivateEye-v5", "function only functional with Montezuma Revenge env."
    ram = env.ale.getRAM()
    item_str = format(ram[65], "8b")
    return dict(
        frame_number=ram[0],
        room_number=ram[3],
        x=ram[42],
        y=ram[43],
        score=10000 * ram[19] + 100 * ram[20] + ram[21],
        life_number=ram[58],
        have_amulet=item_str[7] == "1",
        have_key1=item_str[6] == "1",
        have_key2=item_str[5] == "1",
        have_key3=item_str[4] == "1",
        have_key4=item_str[3] == "1",
        have_sword=item_str[2] == "1",
        have_spare=item_str[1] == "1",
        have_torch=item_str[0] == "1",
    )


def round(input: torch.Tensor, decimals: float) -> torch.Tensor:
    """
    Rounding, but extended to every float.

    :param input: Input tensor
    :type input: torch.Tensor
    :param decimals: Decimals, can be float
    :type decimals: float
    :return: The rounded tensor
    :rtype: torch.Tensor

    Example:
    >>> a
    tensor([0.0000, 0.4000, 0.8000, 1.2000, 1.6000])
    >>> torch.round(a, decimals=0.8)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: round() received an invalid combination of arguments - got (Tensor, decimals=float), but expected one of:
     * (Tensor input, *, Tensor out)
     * (Tensor input, *, int decimals, Tensor out)
    >>> round(a, decimals=0.2)
    tensor([0.0000, 0.6310, 0.6310, 1.2619, 1.8929])
    """
    return torch.round(input * 10**decimals) / 10**decimals


def estimate_density(x: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
    """
    Estimate the density of x within the dataset

    :param x: Points to evaluate density
    :type x: torch.Tensor
    :param dataset: The samples from the distribution to estimate
    :type dataset: torch.Tensor
    :return: The estiamte density on x
    :rtype: torch.Tensor
    """
    n, d = samples.shape
    k = int(2 * n ** (1 / d))
    cdist = torch.cdist(x, samples)
    dist_to_kst = cdist.topk(k, largest=False)[0][:, -1]
    return dist_to_kst ** (-d)

if __name__=='__main__':
    p = 0.0001
    sample_geometric_with_max(p, 2)
