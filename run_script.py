"""
This script is used to run evaluations on a set of elements using the STEER benchmark.
It takes command-line arguments to specify the elements to evaluate, the directory where the elements database is kept,
the directory where the configurations are kept, and whether to run inference on only API models.

The script uses concurrent.futures to parallelize the evaluation process, utilizing all available CPU cores.
For each task, it submits a job to the ThreadPoolExecutor, which runs the `run_evaluation` function on the input path.
As each job completes, the result is printed.

If the `--task-name` argument is set to 'all', the script evaluates all elements in the elements database.
Otherwise, it evaluates the specified task.

Usage:
    python run_script.py --task-name <task_name> [--element-dir <element_dir>] [--config-dir <config_dir>] [-api]

Arguments:
    --task-name: The name of the task(s) to evaluate. Use 'all' to evaluate all elements.
    --element-dir: Path to the directory where the elements database is kept. Default is 'elements/'.
    --config-dir: Path to the directory where the configurations are kept. Default is 'configurations/'.
    -api: Flag that runs inference on only API models.

"""

# Built-in packages
import argparse
import os
import concurrent.futures

# External packages

# Local packages
from utils.inference_utils import run_evaluation
from utils.utils import get_input_paths, dir_path
from utils.logger_utils import JobLogger

CONFIG_DIR = '/var/steer/configurations/'
ELEMENTS_DIR = '/var/steer/elements/'

def main(task_names, api, config_dir, model_name):
    """
    Main function to run evaluations on a set of elements.

    Args:
        task_names (str or list): The name(s) of the task(s) to evaluate. Use 'all' to evaluate all elements.
        api (bool): Flag that runs inference on only API models.
        config_dir (str): Path to the directory where the configurations are kept.

    Returns:
        None
    """
    job_configs = get_input_paths(task_names=task_names, config_dir=config_dir)

    if type(task_names) == str:
        run_evaluation(job_configs['input_path'], api, model_name)
    else:
        max_threads = os.cpu_count() or 1  # Use all available cores, default to 1 if not detected
        # raise NotImplementedError
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            # Submit all jobs to the executor
            future_to_job = {executor.submit(run_evaluation, job_config[1], api): job_config[0] for job_config in job_configs}

            # As each job completes, get the result
            for future in concurrent.futures.as_completed(future_to_job):
                job_id = future_to_job[future]
                try:
                    result = future.result()
                    print(f"Result of job {job_id}: {result}")
                except Exception as exc:
                    print(f"Job {job_id} generated an exception: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Instruction style')
    parser.add_argument('--task-name', '-t', type=str, help="Elements to evaluate")
    parser.add_argument('--model-name', '-m', type=str, help="Name of the model to use for inference", default='all')
    parser.add_argument('--elements-dir', '-e', type=dir_path, nargs='?', default=ELEMENTS_DIR, help="Path to where the elements database is kept")
    parser.add_argument('--config-dir', '-c', type=dir_path, nargs='?', default=CONFIG_DIR, help="Path to where the configurations are kept")
    parser.add_argument('-api', action='store_true', help='Flag that runs inference on only api models')
    args = parser.parse_args()
    print(f'\n\n {args.config_dir} \n\n')
    # TODO: currently only supports single task name 'all' is not supported
    # if args.task_name == 'all':
    #     assert os.path.exists(args.element_dir), 'Path to elements is not at current working directory, supply the directory with --element-dir'
    #     elements = next(os.walk(args.element_dir))[1]
    #     job_loggers = {element: JobLogger('logs', element) for element in elements}
    #     main(elements, args.api, args.config_dir, None)
    if args.model_name == 'all':
        main(args.task_name, args.api, args.config_dir, None)
    else:
        main(args.task_name, args.api, args.config_dir, args.model_name)
