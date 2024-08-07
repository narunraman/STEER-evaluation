# Built-in packages
import os
from collections import defaultdict
import json
from itertools import product
from string import ascii_uppercase
# External packages
import pandas as pd
import torch



def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def get_chat_type(model_name):
    if 'gpt' in model_name.lower() or 'chat' in model_name.lower() or ('mistralai' in model_name.lower() and 'instruct' in model_name.lower()):
        return 'list'
    elif 'falcon' in model_name.lower() and 'instruct' in model_name.lower():
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


def load_dfs(filepath, as_dict = False):
    print(filepath)
    if as_dict:
        return {'questions_df': pd.read_pickle(filepath+'questions.pkl'), 'questions_metadata': pd.read_pickle(filepath+'questions_metadata.pkl'), 'options_df': pd.read_pickle(filepath+'options.pkl'), 'answers_df': pd.read_pickle(filepath+'answers.pkl')}
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


def get_input_paths(task_names: str | list, config_dir: str) -> list[dict[str, str]] | dict[str, str]:
    if type(task_names) == str:
        if not os.path.exists(os.path.join(config_dir, task_names + '.json')):
            print(f'Creating default config file in {config_dir}')
            write_default_config(config_dir, task_names)
        return {'task_name': task_names, 'input_path': config_dir + task_names + '.json'}
    elif type(task_names) == list:
        for task_name in task_names:
            if not os.path.exists(config_dir + task_name + '.json'):
                print('Creating default config file')
                write_default_config(config_dir, task_name)
        return [{'task_name': task_name, 'input_path': config_dir + task_name + '.json'} for task_name in task_names]


def write_default_config(base_dir, task_name):
    config = {
        "task_name": task_name,
        "task_path": f"elements/{task_name}/",
        "num_sample": 500,
        "local": 1,
        "resume_running": 0,
        "model_path": "/home/narunram/scratch/models/",
        "models": {
            "gpt-35-turbo-1106": {},
            "gpt-4-1106-preview": {},
            "gpt-4": {},
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
    

def merge_dfs(questions_df, options_df, answers_df, questions_metadata=None):
    """
    Merge DataFrames containing questions, options, answers, and metadata.

    Args:
        questions_df (pandas.DataFrame): DataFrame containing questions.
        options_df (pandas.DataFrame): DataFrame containing options.
        answers_df (pandas.DataFrame): DataFrame containing answers.
        questions_metadata (pandas.DataFrame): DataFrame containing questions metadata.

    Returns:
        pandas.DataFrame: Merged DataFrame.
    """
    df = pd.merge(options_df, questions_df, on='question_id', how='inner')
    # print(df)
    df = pd.merge(df, answers_df, on=['question_id', 'option_id'])
    if questions_metadata is not None:
        df = pd.merge(df, questions_metadata, on='question_id', how='inner')
    return df

def get_base_ids(questions_df):
    """
    Get unique base IDs from the questions DataFrame.

    Args:
        questions_df (pandas.DataFrame): DataFrame containing questions.

    Returns:
        list: List of unique base IDs.
    """
    return questions_df['question_id'].str.split('_').str[0].unique().tolist()

def get_max_base_id(questions_df):
    """
    Get the maximum base ID from the questions DataFrame.

    Args:
        questions_df (pandas.DataFrame): DataFrame containing questions.

    Returns:
        int: Maximum base ID.
    """
    base_ids = [int(base_id) for base_id in get_base_ids(questions_df)]
    return max(base_ids)


def get_related_questions(questions_df, base_id):
    """
    Get questions related to a specific base ID.

    Args:
        questions_df (pandas.DataFrame): DataFrame containing questions.
        base_id (str): Base ID to filter the questions.

    Returns:
        pandas.DataFrame: DataFrame containing related questions.
    """
    return questions_df[questions_df['question_id'].str.startswith(f"{base_id}_")]


def get_related_options(options_df, base_id, sub_id=None):
    """
    Get options related to a specific base ID.

    Args:
        options_df (pandas.DataFrame): DataFrame containing options.
        base_id (str): Base ID to filter the options.

    Returns:
        pandas.DataFrame: DataFrame containing related options.
    """
    if sub_id:
        return options_df[options_df['question_id'] == f"{base_id}_{sub_id}"]
    return options_df[options_df['question_id'].str.startswith(f"{base_id}_")]

def get_related_answer(answers_df, base_id, sub_id=None):
    
    if sub_id:
        try:
            selected_answers_df = answers_df[answers_df['question_id'] == f"{base_id}_{sub_id}"]['correct'].tolist()
        except KeyError:
            selected_answers_df = answers_df[answers_df['question_id'] == f"{base_id}_{sub_id}"]['correct_answer'].tolist()
        try:
            return ascii_uppercase[selected_answers_df.index(1)]
        except ValueError:
            return ascii_uppercase[selected_answers_df.index(True)]
    curr_answers = answers_df[answers_df['question_id'].str.startswith(f"{base_id}_")].groupby('option_id')
    try:
        return [ascii_uppercase[curr_answers.get_group(option_id)['correct'].tolist().index(1)] for option_id in curr_answers.groups]
    except ValueError:
        return [ascii_uppercase[curr_answers.get_group(option_id)['correct'].tolist().index(True)] for option_id in curr_answers.groups]

def get_unique_questions(questions_df, base_id):
    """
    Get unique questions related to a specific base ID.

    Args:
        questions_df (pandas.DataFrame): DataFrame containing questions.
        base_id (str): Base ID to filter the questions.

    Returns:
        pandas.DataFrame: DataFrame containing unique questions.
    """
    return questions_df[questions_df['question_id'].str.startswith(f"{base_id}_")].drop_duplicates(subset=['question_text'])


def row_to_mcqs(base_id, df):
    """
    Convert a row to multiple-choice question (MCQ) string.

    Args:
        base_id (str): Base ID to convert to MCQ string.
        df (pandas.DataFrame): Merged DataFrame containing questions, options, and answers.
    Returns:
        list: list of MCQ strings.
    """
    questions = get_unique_questions(df, base_id)['question_text'].tolist()
    mcq_string = []
    for i, question in enumerate(questions):
        curr_string = f"{question}\n"
        options = get_related_options(df, base_id, str(i))['option_text'].tolist()
        answer = get_related_answer(df, base_id, str(i))
        for i, option in enumerate(options):
            curr_string += f"{chr(65 + i)}. {option}\n"
        curr_string +=f"\nCorrect Answer: {answer}" 
        mcq_string.append(curr_string)
    return mcq_string
