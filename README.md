# STEER-evaluation

## Description

This repository contains the STEER-evaluation project, which is a benchmarking tool for evaluating the performance of STEER elements.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/narunraman/STEER-evaluation.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Create a configuration json (see configurations/independence_risk.json as an example)

    ```python
    {
        "task_name": <task-name>,
        "task_path": <path-to-elements-directory>,
        "num_sample": <number of questions to sample>,
        "models": {<name-of-local-model|opensource-api-modelname>: {"num_gpus": <number-of-gpus>}},
        "output_path": <save-directory>
    }
    ```

2. Run the evaluation script:

    For api-based models: 
    ```bash
    python run_script.py -t independence_risk -api
    ```

    For local models:
    ```bash
    python run_script.py -t independence_risk
    ```

    The script will default to look at `./configurations/` and `./elements/` directories for files, but you can customize to whichever directory by setting the flags in the command line.
    ```bash
    python run_script.py -t independence_risk -e /path/to/elements/ -c /path/to/configurations
    ```
    

3. View the logs in the `grouped_counts.csv` file in `logs/`


## Contact

For any questions or feedback, please contact the project maintainer:

- Name: Narun Raman
- Email: narunram@cs.ubc.ca
