
"""
Configuration file for NeuPan.

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

import torch
import numpy as np

device = torch.device("cpu")
time_print = False
tensor_dtype= torch.float32

def np_to_tensor(array, requires_grad=False):
        
    if np.isscalar(array):
        output_tensor = torch.tensor(array, dtype=tensor_dtype, requires_grad=requires_grad).to(device)
    else:
        output_tensor = torch.from_numpy(array).type(tensor_dtype).to(device)

    if requires_grad:
        output_tensor.requires_grad_()
        
    return output_tensor

def tensor_to_np(tensor):

    if tensor is None:
        return None

    tensor = tensor.cpu()
    return tensor.detach().numpy()

def value_to_tensor(value, requires_grad=False):
    
    if value is None:
        return None

    return torch.tensor(value, dtype=tensor_dtype, requires_grad=requires_grad).to(device)

def to_device(tensor):
    return tensor.to(device)