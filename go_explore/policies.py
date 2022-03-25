import gym.spaces
import torch as th
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.type_aliases import TensorDict
from torch import nn

from go_explore.cells import CellFactory


class MyCombinedExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces containing keys "observation" and "goal".

    Builds a feature extractor "observation" (CNN or MLP, depending on input shape).
    Use cell_factory for the "goal".
    The output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space: The Dict observation space
    :param cell_factory: The cell factory
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(self, observation_space: gym.spaces.Dict, cell_factory: CellFactory, cnn_output_dim: int = 256):
        super(MyCombinedExtractor, self).__init__(observation_space, features_dim=1)

        self.cell_factory = cell_factory

        extractors = {}
        total_concat_size = 0

        if is_image_space(observation_space["observation"]):
            extractors["observation"] = NatureCNN(observation_space["observation"], features_dim=cnn_output_dim)
            total_concat_size += cnn_output_dim
        else:
            # The observation key is a vector, flatten it if needed
            extractors["observation"] = nn.Flatten()
            total_concat_size += get_flattened_obs_dim(observation_space["observation"])

        extractors["goal"] = self.cell_factory
        total_concat_size += get_flattened_obs_dim(observation_space["goal"])

        self.extractors = extractors

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)
