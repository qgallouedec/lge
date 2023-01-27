from torch import nn


class BaseModule(nn.Module):
    encoder: nn.Module  # A module is expected to contain an encoder.
