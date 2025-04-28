
"""
ObsPointNet is a neural network structure of DUNE model. It maps each obstacle point to the latent distance feature mu.

Developed by Ruihua Han
Copyright (c) 2025 Ruihua Han <hanrh@connect.hku.hk>

NeuPAN planner is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

NeuPAN planner is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with NeuPAN planner. If not, see <https://www.gnu.org/licenses/>.
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