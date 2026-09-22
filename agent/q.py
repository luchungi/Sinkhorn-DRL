"""Q-network over state dicts for discrete actions."""
from typing import Sequence

import torch.nn as nn


class QFunc(nn.Module):
    """Fully connected Q-network with tanh hidden activations.

    Args:
        input_size: Dimension of the flattened state.
        hidden_size: Widths of the hidden layers.
        output_size: Number of discrete actions.
        spec: StateSpec used to flatten state dicts.
    """

    def __init__(self, input_size: int, hidden_size: Sequence[int], output_size: int, spec):
        super().__init__()
        self.spec = spec
        activation_fn = nn.Tanh()
        layers = [nn.Linear(input_size, hidden_size[0])]
        for dim in hidden_size[1:]:
            layers.extend([activation_fn, nn.Linear(layers[-1].out_features, dim)])
        layers.append(activation_fn)
        layers.append(nn.Linear(hidden_size[-1], output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        """Compute Q-values for every action.

        Args:
            state: State dict with any number of leading batch dimensions.

        Returns:
            Tensor of shape (..., output_size).
        """
        return self.net(self.spec.flatten(state))
