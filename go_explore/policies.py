import gym.spaces
import torch as th
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.type_aliases import TensorDict
from torch import nn

from go_explore.cells import CellFactory


class GoExploreExtractor(BaseFeaturesExtractor):
    """
    Feature extraction for GoExplore. Dict observation spaces containing keys "observation" and "goal".

    The output is the concatenation of:
     - the output of a feature extractor on the "observation" (CNN or MLP, depending on input shape).
     - the output of the cell_factory on the "goal".

    The size of the cell may vary during the learning process. However, it is not possible to
    constantly change the network structure. We therefore set the size of the output constant
    equal to the maximum size that the cell can take. We complete the unused outputs with zeros.
    The maximum size that the cell can take is the size of the goal space (which is the same as the
    observation space)

    :param observation_space: The Dict observation space
    :param cell_factory: The cell factory
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(self, observation_space: gym.spaces.Dict, cell_factory: CellFactory, cnn_output_dim: int = 256):
        super(GoExploreExtractor, self).__init__(observation_space, features_dim=1)

        self.cell_factory = cell_factory

        if is_image_space(observation_space["observation"]):
            self.observation_extractor = NatureCNN(observation_space["observation"], features_dim=cnn_output_dim)
            observation_feature_size = cnn_output_dim
        else:
            # The observation key is a vector, flatten it if needed
            self.observation_extractor = nn.Flatten()
            observation_feature_size = get_flattened_obs_dim(observation_space["observation"])

        cell_size = get_flattened_obs_dim(observation_space["goal"])

        # Update the features dim manually
        self._features_dim = observation_feature_size + cell_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        features = th.zeros((observations["observation"].shape[0], self._features_dim))  # (N_ENVS x FEAT_DIM)
        non_zeros_features = th.cat(
            [self.observation_extractor(observations["observation"]), self.cell_factory(observations["goal"])], dim=1
        )
        features[:, : non_zeros_features.shape[1]] = non_zeros_features
        return features
