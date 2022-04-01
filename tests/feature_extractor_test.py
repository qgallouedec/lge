import numpy as np
import torch as th
from gym import spaces

from go_explore.cells import DownscaleObs, ImageGrayscaleDownscale
from go_explore.feature_extractor import GoExploreExtractor


def test_feature_extractor():
    observation_space = spaces.Dict({"observation": spaces.Box(-1, 1, (2,)), "goal": spaces.Box(-1, 1, (2,))})
    cell_factory = DownscaleObs(observation_space)
    feature_extractor = GoExploreExtractor(observation_space, cell_factory)
    observation = {"observation": th.Tensor([[-0.5, 0.5]]), "goal": th.Tensor([[0.0, -0.3]])}
    feature = feature_extractor(observation)
    assert (feature == th.Tensor([-0.5, 0.5, 0.0, -1.0])).all()


def test_feature_extractor_image():
    height = 210
    width = 160
    cnn_output_dim = 64
    observation_space = spaces.Dict(
        {
            "observation": spaces.Box(0, 255, (3, height, width), dtype=np.uint8),
            "goal": spaces.Box(0, 255, (3, height, width), dtype=np.uint8),
        }
    )
    cell_factory = ImageGrayscaleDownscale(height // 5, width // 5, 50)
    feature_extractor = GoExploreExtractor(observation_space, cell_factory, cnn_output_dim)
    observation = {"observation": th.randint(0, 255, (1, 3, height, width)), "goal": th.randint(0, 255, (1, 3, height, width))}
    feature = feature_extractor(observation)
    assert feature.shape == (1, cnn_output_dim + height * width)
    max_used_idx = cnn_output_dim + height // 5 * width // 5
    assert (feature[:, max_used_idx:] == th.zeros_like(feature[:, max_used_idx:])).all()
