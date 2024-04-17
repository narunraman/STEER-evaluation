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
        "models": {<name-of-local-model|opensource-api-modelname>: <number-of-gpus>},
        "output_path": <save-directory>
    }
    ```

2. Run the benchmarking script:
    
    Currently only supports api-based models:

    ```bash
    python run_script.py -t independence_risk -api
    ```

    NOTE: the script will look at the `configurations/` and `elements/` directories for files, but can customize to whichever directory by setting the flags in the command line.

3. View the logs in the generated `grouped_counts.csv` file in `logs/`


## Contact

For any questions or feedback, please contact the project maintainer:

- Name: Narun Raman
- Email: narunram@cs.ubc.ca
