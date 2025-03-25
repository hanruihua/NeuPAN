
"""
ObsPointNet is a neural network structure of DUNE model. It maps each obstacle point to the latent distance feature mu.

Developer: Han Ruihua <hanrh@connect.hku.hk>
"""

import torch.nn as nn
import torch

class ObsPointNet(nn.Module):
    def __init__(self, input_dim: int = 2,  output_dim: int=4) -> None:
        super(ObsPointNet, self).__init__()

        hidden_dim = 32

        self.MLP = nn.Sequential(   nn.Linear(input_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.Tanh(),
                                    nn.Linear(hidden_dim, output_dim),
                                    nn.ReLU(),
                                    )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.MLP(x)