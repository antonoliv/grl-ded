from typing import Dict

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class GraphExtractor(BaseFeaturesExtractor):
    """
    Graph Features extractor for stable-baselines3 models.
    It filters the observation space to keep only the feature matrix as input for the model.
    """

    def __init__(self, observation_space: spaces.Dict) -> None:
        """
        Initialize the GraphExtractor object.
        :param observation_space: dict space containing
        """
        super().__init__(observation_space, features_dim=1)

        if observation_space.spaces["x"] is None:
            raise ValueError("No 'x' key in the observation space")

        self.extractor = nn.Flatten()

        # Update the features dim manually
        self._features_dim = spaces.flatdim(observation_space.spaces["x"])

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        """
        Forward pass in the GraphExtractor.

        :param observations: batch of observations
        :return: extracted features tensor
        """
        return self.extractor(observations["x"])
