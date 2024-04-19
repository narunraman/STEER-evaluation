# Built-in packages
import os
from collections import defaultdict
import json
from itertools import product
from string import ascii_uppercase
# External packages
import pandas as pd
import torch

def get_chat_type(model_name):
    if 'gpt' in model_name.lower() or 'Llama-2-7b-chat-hf' in model_name.lower() or 'Mistral-7B' in model_name.lower():
        return 'list'
    elif 'falcon-7b-instruct' in model_name.lower():
        return 'textual'
    return None

def normalize_dict(probs):
    total = sum(probs.values())
    return {prob: probs[prob]/total for prob in probs}

def get_option_letter(question_num, option_num, num_options):
    return (question_num * num_options) + option_num

def get_option_letters(options_lst):
    it = iter(ascii_uppercase)
    return [list(next(it) for _ in sublist) for sublist in options_lst]

def get_gpu_memory(max_gpus=None):
    """Get available memory for each GPU."""
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024 ** 3)
            allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


def load_dfs(filepath):
    return pd.read_pickle(filepath+'questions.pkl'), pd.read_pickle(filepath+'questions_metadata.pkl'), pd.read_pickle(filepath+'options.pkl'), pd.read_pickle(filepath+'answers.pkl')

def flatten_list(matrix):
    return [item for row in matrix for item in row]

def read_as_defaultdict(json_path: str, to_return = None):

    data = defaultdict(lambda: to_return)
    with open(json_path, 'r') as f:
        old_data = json.load(f)

    for key in old_data:
        data[key] = old_data[key]
    return data

def print_chat(messages):
    for message in messages:
        print(f"{message['role'].upper():}", message['content'])


def get_input_paths(task_names, config_dir):
    if type(task_names) == str:
        if not os.path.exists(config_dir + task_names + '.json'):
            print('Creating default config file')
            write_default_config(config_dir, task_names)
        return [(task_names, config_dir + task_names + '.json')]
    elif type(task_names) == list:
        for task_name in task_names:
            if not os.path.exists(config_dir + task_name + '.json'):
                print('Creating default config file')
                write_default_config(config_dir, task_name)
        return [(task_name, config_dir + task_name + '.json') for task_name in task_names]


def write_default_config(base_dir, task_name):
    config = {
        "task_name": task_name,
        "task_path": f"elements/{task_name}/",
        "num_sample": 500,
        "local": 1,
        "resume_running": 0,
        "model_path": "/home/narunram/scratch/models/",
        "models": {
            "gpt-35": {},
            "gpt-4": {}
        },
        "output_path": f"results/{task_name}/"
    }

    with open(base_dir + task_name + '.json', 'w') as f:
        json.dump(config, f, indent = 4) 

class ParameterGrid:
    def __init__(self, param_grid):
        # Allow for either a single dictionary or a list of dictionaries
        if isinstance(param_grid, dict):
            self.param_grids = [param_grid]
        elif isinstance(param_grid, list):
            self.param_grids = param_grid
        else:
            raise ValueError("Parameter grid must be a dict or a list of dicts")

        # Ensure that all values in each parameter grid are lists
        for grid in self.param_grids:
            for key in grid:
                if not isinstance(grid[key], list):
                    grid[key] = [grid[key]]

    def __iter__(self):
        # Iterate through each grid and yield the cartesian product of parameters
        for grid in self.param_grids:
            keys = grid.keys()
            for values in product(*grid.values()):
                yield dict(zip(keys, values))

    def __len__(self):
        # Sum of products of lengths of all lists in all grids
        total_length = 0
        for grid in self.param_grids:
            from functools import reduce
            from operator import mul
            product_length = reduce(mul, (len(v) for v in grid.values()), 1)
            total_length += product_length
        return total_length