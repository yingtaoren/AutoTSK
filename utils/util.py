import torch
import logging
import random
import numpy as np

def revise_alpha(alpha):
    threshold = 0.1  # You can adjust the threshold value as needed
    revised_alpha = (alpha > threshold).int()
    return revised_alpha


def check_tensor(tensor, dtype):
    return torch.tensor(tensor, dtype=dtype)


def reset_params(model):

    if hasattr(model, "reset_parameters"):
        model.reset_parameters()
    else:
        for layer in model.children():
            reset_params(layer)


class NumpyDataLoader:

    def __init__(self, *inputs):
        self.inputs = inputs
        self.n_inputs = len(inputs)

    def __len__(self):
        return self.inputs[0].shape[0]

    def __getitem__(self, item):
        if self.n_inputs == 1:
            return self.inputs[0][item]
        else:
            return [array[item] for array in self.inputs]

def add_result(result, model_name, value):
    if model_name not in result:
        result[model_name] = []
        result[model_name].append(value)
    else:
        result[model_name].append(value)

def save_results(results):
    for key, values in results.items():
        
        if len(values) == 0:
            average_value = sum(values) / 1
        else:
            average_value = sum(values) / len(values)
        print(key, values, average_value)
        logging.warning(
            f"model_name: {key}, value: {values}, mean_accuracy: {average_value:.4f}"
        )

def print_results(results):
    for key, values in results.items():
        if len(values) == 0:
            average_value = sum(values) / 1
        else:
            average_value = sum(values) / len(values)

        print(
            f"model_name: {key}, value: {values}, mean_accuracy: {average_value:.4f}"
        )

def set_seed(seed):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
