import pandas as pd

def create_base_results_df():
    results_df = pd.DataFrame(columns = [
            'model',
            'task_name',
            'question_id', 
            'domain',
            'difficulty_level',
            'type',
            'model_answer', 
            'model_explanation', 
            'allow_explanation', 
            'probabilities', 
            'num_shots'])
    return results_df

def create_base_metadata_df():
    metadata_df = pd.DataFrame(columns = [
                    'task_name',
                    'model_name',
                    'question_id',
                    'permutation',
                    'prompt',
                    'inference_time'
    ])
    return metadata_df

def load_results(file_path):
    """
    Load dataset from a file.
    If the file does not exist, return a blank pandas dataframe.
    """
    try:
        return pd.read_pickle(file_path)
    except FileNotFoundError:
        return create_base_results_df()

def load_metadata(file_path):
    """
    Load metadata from a file.
    If the file does not exist, return a blank pandas dataframe.
    """
    try:
        return pd.read_pickle(file_path)
    except FileNotFoundError:
        return create_base_metadata_df()

def save_dataset(dataset, file_path):
    """
    Save dataset to a file.
    """
    dataset.to_pickle(file_path)


def check_num_rows(dataset, args):
    """
    Check if the number of saved input rows for each type, domain, and difficulty_level
    are the same as in args['num_sample'].
    """
    if len(dataset) == 0:
        return False
    num_sample = args['num_sample']
    num_rows = dataset.groupby(['domain', 'difficulty_level', 'type', 'num_shots', 'allow_explanation']).size()

    return all(num_rows == num_sample)


def get_uncompleted_dfs(datasets, question_ids):
    """
    Return the datasets with all uncompleted rows based on question_ids.
    """
    if not question_ids:
        return datasets
    return [dataset[~dataset['question_id'].isin(question_ids)] for dataset in datasets]

