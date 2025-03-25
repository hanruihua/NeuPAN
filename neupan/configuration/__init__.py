import torch
import numpy as np

device = torch.device("cpu")
time_print = False

def np_to_tensor(array):
        
    if np.isscalar(array):
        return torch.tensor(array).type(torch.float32).to(device)

    return torch.from_numpy(array).type(torch.float32).to(device)

def tensor_to_np(tensor):

    if tensor is None:
        return None

    tensor = tensor.cpu()
    return tensor.detach().numpy()

def value_to_tensor(value, requires_grad=False):
    
    if value is None:
        return None

    return torch.tensor(value, dtype=torch.float32, requires_grad=requires_grad).to(device)

def to_device(tensor):
    return tensor.to(device)