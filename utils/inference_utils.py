#Built-in packages
import time
import random
import os
from tqdm import tqdm
from string import ascii_lowercase, ascii_uppercase
# External packages
import numpy as np
import pandas as pd
import openai
import torch
import backoff  # for exponential backoff
# Local packages
from utils.response_utils import parse_response, compute_ece, get_correct, get_true_labels, get_random_acc
from utils.api_response_utils import OPENAI_RESPONSE
from utils.hf_response_utils import get_explanation_probs, HF_RESPONSE
from utils.question_utils import reconstruct_context, build_prefix, get_test_questions, permute_answer, convert_probabilities, append_question
from utils.parsing_utils import find_answer_letter
from utils.utils import get_option_letters, print_chat, read_as_defaultdict, load_dfs, get_chat_type, ParameterGrid
from utils.model_utils import GPTClient, MODEL_PATH, load_model_tokenizer
from utils.logger_utils import JobLogger
from utils.dataset_utils import load_results, load_metadata, check_num_rows, get_uncompleted_dfs

QUESTIONS_DF, QUESTIONS_METADATA, OPTIONS_DF, ANSWERS_DF = None, None, None, None

OPTIONS = list(ascii_uppercase)
LETTERS = list(ascii_lowercase)

USE_LOGPROBS = True

###########################################
##                                       ##
##             Helper Code               ##
##                                       ##
###########################################


def exponential_backoff_decorator(max_retries, base_delay):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    result_func = func(*args, **kwargs)
                    return result_func
                except Exception as e:
                    print(f"Attempt {retries + 1} failed: {e}")
                    retries += 1
                    delay = (base_delay * 2 ** retries + random.uniform(0, 1))
                    print(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
            # raise Exception("Max retries reached, operation failed.")
            print("Max retries reached, operation failed.")

        return wrapper

    return decorator

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_response(client, model, prefix, questions, question_type, options_lst, chat_type):
    outputs =  []
    parsed_results = []
    global USE_LOGPROBS

    for i, question in enumerate(questions):
        context = reconstruct_context(prefix, questions[:i], outputs, chat_type)

        if len(context) > 1:
            context.append({'role': 'user', 'content': question})
        elif len(context) == 1:
            context[0]['content'] += '\n' + question
        else:
            context.append({'role': 'user', 'content': question})

        # NOTE: until we get access to tokenizer can't do mc-separate
        if question_type == 'mc':
            
            answer, probs, has_logprobs = OPENAI_RESPONSE[question_type](client=client, 
                        valid_tokens = get_option_letters(options_lst)[i],
                        model = model, 
                        context = context, 
                        use_logprobs = USE_LOGPROBS, 
                        num_logprobs = 5)
            USE_LOGPROBS = has_logprobs
            
            parsed_results.append(['', answer, probs])

            outputs.append(answer)
        
        # Model is allowed to explain and answer
        elif question_type == 'explanation':
            output = client.get_explanation(model = model, messages = context, max_tokens = None)
            explanation, answer = parse_response(output, options_lst, i)
            outputs.append(output)
            valid_tokens = get_option_letters(options_lst)[i]
            parsed_results.append([explanation, answer, {valid_token: 0.0 for valid_token in valid_tokens}])

        # Model is allowed to explain only
        elif i % 2 == 0 and (question_type == 'sequential-hidden' or question_type == 'sequential-shown'):
            output = client.get_explanation(model = model, messages = context, max_tokens = None)
            outputs.append(output)

        # Model is allowed to answer only
        elif i % 2 == 1 and (question_type == 'sequential-hidden' or question_type == 'sequential-shown'):
            answer, probs, has_logprobs = OPENAI_RESPONSE['mc'](client=client,
                        model = model,
                        valid_tokens = get_option_letters(options_lst)[i//2],
                        context = context,
                        use_logprobs = USE_LOGPROBS,
                        num_logprobs = 5)
            USE_LOGPROBS = has_logprobs
            outputs.append(answer)
            parsed_results.append([outputs[i-1], answer, probs])

        else:
            raise ValueError(f"Invalid question_type: {question_type}")
    
    return np.array(parsed_results).T.tolist()

def get_response_hf(model, tokenizer, device, prefix, questions, question_type, options_lst, chat_type):
    outputs = []
    parsed_results = []

    option_letters = get_option_letters(options_lst)

    for i, question in enumerate(questions):
        # context is either a list of dictionaries or a string depending on chat_type
        context = reconstruct_context(prefix, questions[:i], outputs, chat_type)
        
        prompt = append_question(context, question, chat_type)

        if question_type == 'mc' or question_type == 'mc-separate':
            answer, probs = HF_RESPONSE[question_type](model, tokenizer, prompt, option_letters[i], device, chat_type)
            outputs.append(answer)
            parsed_results.append(['', answer, probs])

        elif question_type == 'explanation':
            output = HF_RESPONSE[question_type](model, tokenizer, prompt, device, chat_type)
            outputs.append(output)
            answer_letter = find_answer_letter(output)
            
            new_output, probs = get_explanation_probs(model, tokenizer, context, output, option_letters[i], device, chat_type)

            # TODO: should check if new_output is the same as answer_letter

            parsed_results.append([output, answer_letter, probs])
        
        elif i % 2 == 0 and (question_type == 'sequential-hidden' or question_type == 'sequential-shown'):
            output = HF_RESPONSE['explanation'](model, tokenizer, prompt, device, chat_type)
            outputs.append(output)

        elif i % 2 == 1 and (question_type == 'sequential-hidden' or question_type == 'sequential-shown'):
            answer, probs = HF_RESPONSE['mc'](model, tokenizer, prompt, options_lst[i], device, chat_type)
            outputs.append(answer)
            parsed_results.append([outputs[i-1], answer, probs])
        

        # TODO: add condition where the model is asked if the answer is correct or not
    return np.array(parsed_results, dtype=object).T.tolist()


#########################################################################################
#########################################################################################
###                                                                                   ###
###                                                                                   ###
###                                                                                   ###
###                             Run Inference Code                                    ###
###                                                                                   ###
###                                                                                   ###
###                                                                                   ###
#########################################################################################
#########################################################################################





def create_results_dict(params, task_name, model_name, base_id, sub_id, permuted_answer, model_answer, model_explanation, probabilities, permutations):
    # Setup results_dict
    results = {key: value for key, value in params.items()}
    results['task_name'] = task_name
    results['model'] = model_name
    results['question_id'] = f'{base_id}_{sub_id}'
    results['permuted_answer'] = permuted_answer
    results['model_answer'] = model_answer
    results['model_explanation'] = model_explanation
    results['probabilities'] = probabilities
    results['accuracy'] = get_correct(results['question_id'], ANSWERS_DF, permuted_answer)
    # TODO: when expanding to multiple part questions, need to update get_random_acc to take in task_name and questions_metadata
    results['normalized_accuracy'] = results['accuracy'] - get_random_acc(results['question_id'], ANSWERS_DF)
    
    true_labels = get_true_labels(results['question_id'], ANSWERS_DF)
    probabilities_list = convert_probabilities(probabilities, sub_id, permutations)
    results['expected_calibration'] = compute_ece(np.array(probabilities_list), np.array(true_labels))
    
    return results


# setup parameter grid
def setup_param_grid(args, api=False):
    params = []
    # if there is no allow_explanation, default to both adaptation regimes
    if 'allow_explanation' not in args:
        args['allow_explanation'] = 2
    if args['allow_explanation'] % 2 == 0:
        if api:
            params.append({
                'num_shots': args.get('num_shots', [0, 1, 2, 5]), 
                'allow_explanation': [False], 
                'question_type': [
                    'mc'
                ],
                'robust-calibration': [False, True],
                'num_sample': [args['num_sample']]
            })
        else:
            params.append({
                    'num_shots': args.get('num_shots', [0, 1, 2, 5]), 
                    'allow_explanation': [False], 
                    'question_type': [
                        'mc',
                        'mc-separate'
                    ],
                    'num_sample': [args['num_sample']]
                }) 
    if args['allow_explanation'] > 0:
        params.append({
                'num_shots': args.get('num_explanations', [0, 1, 2, 5]), 
                'allow_explanation': [True], 
                'question_type': [
                    'explanation',
                    'sequential-hidden', 
                    'sequential-shown'
                ], 
                'robust-calibration': [False, True],
                'num_sample': [args['num_sample']]
            })
    return ParameterGrid(
        params
    )

# TODO: do this
# def eval_async(args, api, model_name, client = 'gpt'):
#     job_logger = JobLogger(f"/var/steer/logs/{args['task_name']}/{model_name}/", api)
#     progress_bar = job_logger.tqdm

#     results_path = os.path.join(args['output_path'], model_name, args['task_name'] + '_results.pkl')
#     metadata_path = os.path.join(args['output_path'], model_name, args['task_name'] + '_metadata.pkl')

#     results_df = load_results(results_path)
#     if check_num_rows(results_df, args):
#         job_logger.log_output(f"Model {model_name} has already been evaluated.")
#         continue
#     results_metadata_df = load_metadata(metadata_path)

#     # TODO: handle logic for different clients
#     client = GPTClient()

#     param_grid = setup_param_grid(args, api=True)

#     try:
#         sampled_df = test_metadata.groupby(['type', 'domain', 'difficulty_level']).sample(n=params['num_sample'], random_state=42)
#     except ValueError: 
#         sampled_df = test_metadata
#     sampled_qids = set(sampled_df['question_id'])






def construct_robust_calibrated_distribution(questions_metadata, options_df):
    '''
    Create a boolean list of whether a question should be calibrated or not.
    For each group of (domain, type, difficulty_level) we want to calibrate 1/num_options questions.
    '''

    robust_calibration_mask = [False] * len(questions_metadata)
    
    # Group by (domain, type, difficulty_level)
    grouped = questions_metadata.groupby(['domain', 'type', 'difficulty_level'])
    
    for _, group in grouped:
        num_options = len(options_df.query("question_id == @group.iloc[0]['question_id']"))
        if num_options == 0:
            raise ValueError(f"Question {group.iloc[0]['question_id']} has no options.")
        if num_options == 1:
            continue
        
        # Calculate the number of questions to calibrate
        num_to_calibrate = max(1, len(group) // num_options)
        
        # Randomly select questions to calibrate
        indices_to_calibrate = random.sample(group.index.tolist(), num_to_calibrate)
        
        for idx in indices_to_calibrate:
            robust_calibration_mask[idx] = True
    
    return robust_calibration_mask


def eval_models(args, api, device=None, model_name = None):


    if model_name:
        model_names = [model_name]
    else:
        model_names = list(args['models'].keys())
    # save results per model
    for model_name in model_names:
        if api:
            job_logger = JobLogger(f"/var/steer/logs/{args['task_name']}/{model_name}/", api)
            progress_bar = job_logger.tqdm

            results_path = os.path.join(args['output_path'], model_name, args['task_name'] + '_results.pkl')
            metadata_path = os.path.join(args['output_path'], model_name, args['task_name'] + '_metadata.pkl')
        else:
            job_logger = JobLogger(f"/home/narunram/scratch/llm_outputs/{model_name.split('--')[-1].title()}/", api)
            progress_bar = tqdm

            results_path = os.path.join(args['output_path'], model_name.split('--')[-1].title(), args['task_name'] + '_results.pkl')
            metadata_path = os.path.join(args['output_path'], model_name.split('--')[-1].title(), args['task_name'] + '_metadata.pkl')

        results_df = load_results(results_path)
        if check_num_rows(results_df, args):
            job_logger.log_output(f"Model {model_name} has already been evaluated.")
            continue
        results_metadata_df = load_metadata(metadata_path)
        
        # Load model and tokenizer
        if device:
            num_gpus = torch.cuda.device_count()

            job_logger.log_output(f"Loading model: {model_name}")
            model, tokenizer = load_model_tokenizer(MODEL_PATH, model_name, device, num_gpus)
            if not model:
                continue
            else:
                job_logger.log_output(f'Model {model_name} loaded')
            
            try:
                for param in model.parameters():
                    job_logger.log_output(param.device)
            except:
                pass
        else:
            client = GPTClient()
        
        
        # Setup parameter grid
        param_grid = setup_param_grid(args, api)
        
        

        # Running inference
        for params in param_grid:
            # Get all questions that have not been answered under the current parameters (num_shots, allow_explanation, etc.)
            curr_results = results_df.query("num_shots == @params['num_shots'] and allow_explanation == @params['allow_explanation']")
            questions_df, questions_metadata, options_df, answers_df = get_uncompleted_dfs([QUESTIONS_DF, QUESTIONS_METADATA, OPTIONS_DF, ANSWERS_DF], curr_results['question_id'].tolist())
            # Get all questions that are not explanations by question_id
            question_ids = questions_df.query("explanation == False")['question_id']
            test_metadata = questions_metadata.query("question_id in @question_ids")
            # test_metadata = questions_metadata.iloc[questions_df.query("explanation == False").index]
            try:
                sampled_df = test_metadata.groupby(['type', 'domain', 'difficulty_level']).sample(n=params['num_sample'], random_state=42)
            except ValueError: 
                sampled_df = test_metadata
            sampled_qids = set(sampled_df['question_id'])

            # Filtered DataFrames based on sampled question_ids
            filtered_questions_df = questions_df.query("question_id in @sampled_qids")
            filtered_options_df = options_df.query("question_id in @sampled_qids")

            base_ids = set(question_id.split('_')[0] for question_id in sampled_qids)
            filtered_questions_metadata = questions_metadata.query("question_id in @sampled_qids")
            robust_calibration_mask = construct_robust_calibrated_distribution(filtered_questions_metadata, filtered_options_df)

            # iterate over questions by id
            for run_num, base_id in enumerate(progress_bar(base_ids, desc=str(params), dynamic_ncols=True)):
                # build prefix for few-shot prompting
                task_data = questions_metadata.query(f"question_id == '{base_id}_0'").to_dict('records')[0]
                try:
                    prefix = build_prefix(task_data, questions_df, options_df, questions_metadata, answers_df, params)
                except ValueError as e:
                    job_logger.log_output(f"Error: {e}. Skipping {args['num_shots']} num_shots for {task_data}.")
                    continue

                # build question string
                if params['robust-calibration']:
                    test_questions, test_options, permutations = get_test_questions(base_id, filtered_questions_df, filtered_options_df, params, answers_df, replace_answer = robust_calibration_mask[run_num])
                else:
                    test_questions, test_options, permutations = get_test_questions(base_id, filtered_questions_df, filtered_options_df, params)

                # Track total time to run inference on a model
                start_time = time.time()

                # Get model answer for question
                if device:
                    model_explanations, model_answers, probabilities = get_response_hf(
                        model=model, 
                        tokenizer=tokenizer, 
                        device=device, 
                        prefix=prefix, 
                        questions=test_questions, 
                        question_type=params['question_type'],
                        options_lst=test_options,
                        chat_type=get_chat_type(model_name)
                    )
                else:
                    try:
                        model_explanations, model_answers, probabilities = get_response(
                            client=client, 
                            model=model_name,
                            prefix=prefix, 
                            questions=test_questions, 
                            question_type=params['question_type'],
                            options_lst=test_options,
                            chat_type=get_chat_type(model_name)
                        )
                    except openai.BadRequestError as e:
                        job_logger.log_error(f"Error: {e}")
                        continue


                inference_time = time.time() - start_time

                # Reverse permutation answer
                # Permuted answers are the index value into the list of options
                permuted_answers = permute_answer(model_answers=model_answers, permutations=permutations)

                # Instantiate the results dataframe: num_shots, allow_explanation, etc.
                result_params = params.copy()
                # This is domain, type, difficulty_level
                for data in task_data:
                    result_params[data] = task_data[data]
                result_params.pop('num_sample')
                # Store results
                results = [create_results_dict(
                    params = result_params,
                    task_name = args['task_name'],
                    model_name = model_name,
                    base_id = base_id,
                    sub_id = i,
                    permuted_answer = permuted_answer,
                    model_answer = model_answers[i],
                    model_explanation = model_explanations[i],
                    probabilities = probabilities[i],
                    permutations = permutations
                ) for i, permuted_answer in enumerate(permuted_answers)]

                results_df = pd.concat([results_df, pd.DataFrame.from_records(results)], sort=False, ignore_index=True)
                
                result_metadata = [{
                    'task_name': args['task_name'],
                    'model_name': model_name,
                    'question_id': f"{base_id}_{i}",
                    "num_shots": result_params['num_shots'],
                    "question_type": result_params['question_type'],
                    'permutation': permutation,
                    'prompt': test_questions[i],
                    'inference_time': inference_time
                    } for i, permutation in enumerate(permutations)]
                results_metadata_df = pd.concat([results_metadata_df if not results_metadata_df.empty else None, pd.DataFrame.from_records(result_metadata)], sort=False, ignore_index=True)

                if run_num % 10 == 0 and job_logger is not None:
                    try:
                        job_logger.log_groupby_counts(results_df if not results_df.empty else None, ['domain', 'difficulty_level', 'type', 'num_shots', 'allow_explanation'])
                    except Exception as e:
                        job_logger.log_output(str(e))

                if not os.path.exists(os.path.dirname(results_path)):
                    os.makedirs(os.path.dirname(results_path))
                if run_num % 100 == 0:
                    results_df.to_pickle(results_path)
                    results_metadata_df.to_pickle(metadata_path)
        # Save per model
        results_df.to_pickle(results_path)
        results_metadata_df.to_pickle(metadata_path)


def run_evaluation(input_path: str, api: bool, model_name: str):
    args = read_as_defaultdict(input_path)

    global QUESTIONS_DF, QUESTIONS_METADATA, OPTIONS_DF, ANSWERS_DF
    QUESTIONS_DF, QUESTIONS_METADATA, OPTIONS_DF, ANSWERS_DF = load_dfs(args['task_path'])


    if api:
        eval_models(args, api, model_name)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('device:', device)
        with torch.inference_mode():
            eval_models(args, False, device, model_name)
