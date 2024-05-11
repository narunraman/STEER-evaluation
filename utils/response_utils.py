# Built-in packages
from string import ascii_uppercase
import re
from collections import defaultdict
# External packages
import torch
import numpy as np
import openai
# Local packages
from utils.parsing_utils import remove_answer_letter
from utils.utils import get_option_letter, normalize_dict

#########################################################################################
#########################################################################################
###                                                                                   ###
###                                                                                   ###
###                                                                                   ###
###                       HuggingFace Model Response Code                             ###
###                                                                                   ###
###                                                                                   ###
###                                                                                   ###
#########################################################################################
#########################################################################################


def normalize_probs(prob_dict):
    """
    Normalizes the probabilities in a dictionary and returns them as a sorted list.

    This function takes a dictionary where the keys are uppercase letters and the values are their associated probabilities. It normalizes these probabilities such that they sum to 1, if the total sum of the values is not zero. If the total sum is zero, indicating that all probabilities are zero or the dictionary is empty, the function will return an empty list. The resulting probabilities are then returned as a list, sorted in ascending order based on their keys.

    Parameters:
    - prob_dict (dict): A dictionary where the keys are categories and the values are probabilities or counts. The keys are expected to be sortable (e.g., strings, numbers).

    Returns:
    - list: A list of the normalized probabilities, ordered according to the sorted keys of the input dictionary.
      If the total sum of the probabilities is zero, an empty list is returned.
    """
    # Calculate the sum of all probability values in the dictionary
    total = sum(prob_dict.values())
    
    # Check if the total sum is zero, and return an empty list if so
    if total == 0:
        total = 1
    
    # Normalize each probability by dividing it by the total sum
    for key in prob_dict:
        prob_dict[key] /= total
    
    # Return a list of the normalized probabilities, sorted by the keys
    return [prob_dict[key] for key in sorted(prob_dict.keys())]


def get_mc_separate(model, tokenizer, text, option_letters, device, chat_type):
    """
    Process a multiple-choice question by appending each option letter and getting the probability.

    Parameters:
    - model: The loaded HuggingFace model.
    - tokenizer: The corresponding tokenizer for the model.
    - text: The MCQ text as a string.
    - option_letters: A list of strings representing the option letters.

    Returns:
    - A dict with 'responses' containing the model's output for each option,
      and 'normalized_log_odds' containing the normalized log odds of the options.
    """
    model.eval()
    option_probs = {}
    with torch.no_grad():
        for option in option_letters:
            if chat_type == 'list':
                option_text = {'role': 'user', 'content': text[-1]['content'] + option}
                messages = text[:-1] + [option_text]
                option_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
            else:
                option_text = f"{text} {option}"
                option_ids = tokenizer.encode(option_text, return_tensors="pt").to(device)
            outputs = model(option_ids)

            # Get logits of the last token produced for each option
            logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Option is a single character and getting its probability
            option_id = tokenizer.encode(option, add_special_tokens=False)[-1]
            option_prob = probs[0, option_id]

            option_probs[option] = option_prob.item()
    
    return max(option_probs, key=option_probs.get), normalize_dict(option_probs)

    

def get_mc(model, tokenizer, text, option_letters, device, chat_type):
    """
    Inspects the full distribution of the next tokens to select only those that are possible option letters.

    Parameters:
    - model: The loaded HuggingFace model.
    - tokenizer: The corresponding tokenizer for the model.
    - text: The MCQ text as a string.
    - option_letters: A list of strings of the option letters.

    Returns:
    - A dict with 'probabilities' containing the probability of each option's initial token,
      and 'normalized_log_odds' containing the normalized log odds of these probabilities.
    """
    model.eval()
    with torch.no_grad():
        # Tokenize the input text
        if chat_type == 'list':
            input_ids = tokenizer.apply_chat_template(text, return_tensors="pt").to(device)
        else:
            input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

        # Get the model's output
        outputs = model(input_ids)

        # Get logits of the next token
        logits = outputs.logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)

    # Map each option letter to its initial token and extract probability
    option_probs = {}
    for option in option_letters:
        option_token_id = tokenizer.encode(option, add_special_tokens=False)[0]
        option_prob = probs[0, option_token_id].item()
        option_probs[option] = option_prob

    return max(option_probs, key=option_probs.get), normalize_dict(option_probs)

def get_mc_option(model, tokenizer, text, options, device, chat_type):
    """
    Inspects the full distribution of the next tokens to select only those that are possible options.

    Parameters:
    - model: The loaded HuggingFace model.
    - tokenizer: The corresponding tokenizer for the model.
    - text: The MCQ text as a string.
    - options: A list of strings representing the initial tokens of the options.

    Returns:
    - A dict with 'probabilities' containing the probability of each option's initial token,
      and 'normalized_log_odds' containing the normalized log odds of these probabilities.
    """
    model.eval()
    with torch.no_grad():
        # Tokenize the input text
        if chat_type == 'list':
            input_ids = tokenizer.apply_chat_template(text, return_tensors="pt").to(device)
        else:
            input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

        # Get the model's output
        outputs = model(input_ids.cuda())

        # Get logits of the next token
        logits = outputs.logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)

    # Map each option to its tokens and extract probability of the entire option text
    option_probs = {}
    for option in options:
        if type(option) != str:
            option = str(option)
        option_token_ids = tokenizer.encode(option, add_special_tokens=False)
        option_text = tokenizer.decode(option_token_ids, skip_special_tokens=True)
        option_prob = probs[0, option_token_ids].prod().item()
        option_probs[option_text] = option_prob

    return max(option_probs, key=option_probs.get), normalize_dict(option_probs)


def get_explanation(model, tokenizer, text, device, chat_type, max_tokens=512):
    """
    Generates a response up to 512 tokens, using greedy sampling (temperature=0).

    Parameters:
    - model: The loaded HuggingFace model.
    - tokenizer: The corresponding tokenizer for the model.
    - text: The starting text to generate from.
    - max_tokens: The maximum length of the token sequence to generate.

    Returns:
    - A string containing the generated sequence.
    """
    # Encode the input text
    if chat_type == 'list':
        input_ids = tokenizer.apply_chat_template(text, return_tensors="pt").to(device)
    else:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    
    # Ensure the model is in evaluation mode
    model.eval()

    # Generate tokens with greedy sampling
    with torch.no_grad():
        greedy_output = model.generate(input_ids, max_length=max_tokens, temperature=0, do_sample=False).to(device)

    # Decode and return the generated text
    generated_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)

    # answer_letter = find_answer_letter(generated_text)
    # if answer_letter is None:
    return generated_text


def get_explanation_probs(model, tokenizer, context, text, option_letters, device, chat_type):
    new_text = remove_answer_letter(text)
    if chat_type == 'list':
        prompt = context + {'role': 'assistant', 'content': new_text}
    elif chat_type == 'textual':
        prompt += f"Falcon: {new_text}"
    else:
        prompt = context + '\n' + new_text
    new_output, probs = get_mc(model, tokenizer, prompt, option_letters, device, chat_type)
    return new_output, probs

HF_RESPONSE = {
    'mc': get_mc,
    'mc-option': get_mc_option,
    'mc-separate': get_mc_separate,
    'explanation': get_explanation,
}

#########################################################################################
#########################################################################################
###                                                                                   ###
###                                                                                   ###
###                                                                                   ###
###                           OpenAI Model Response Code                              ###
###                                                                                   ###
###                                                                                   ###
###                                                                                   ###
#########################################################################################
#########################################################################################


def get_mc_openai(client, valid_tokens, context, model, use_logprobs, num_logprobs):
    if use_logprobs:
        try:
            answer, probs = client.get_answer(valid_tokens = valid_tokens, model=model, messages = context, max_tokens = 1, logprobs = True, top_logprobs=num_logprobs)
            return answer, probs, use_logprobs
        except openai.BadRequestError as e:
            if "This model does not support the 'logprobs' parameter." in str(e):
                has_logprobs = False
                answer = client.get_answer(valid_tokens = valid_tokens, model=model, messages = context, max_tokens = 1)
                return answer, {valid_token: 0.0 for valid_token in valid_tokens}, has_logprobs
            else:
                raise e
    else:
        answer = client.get_answer(valid_tokens = valid_tokens, model=model, messages = context, max_tokens = 1)
        return answer, {valid_token: 0.0 for valid_token in valid_tokens}, use_logprobs


OPENAI_RESPONSE = {
    'mc': get_mc_openai,
}

#########################################################################################
#########################################################################################
###                                                                                   ###
###                                                                                   ###
###                                                                                   ###
###                            Parsing Output Code                                    ###
###                                                                                   ###
###                                                                                   ###
###                                                                                   ###
#########################################################################################
#########################################################################################


def is_option(answer, options):
    """
    Determines if the given answer is in the list of options.

    This function compares the provided answer against a list of possible options. If the answer matches
    any of the options, it returns the corresponding letter from the ASCII uppercase sequence. Otherwise, it returns False.

    Parameters:
    answer (str): The answer to be checked against the options.
    options (list): A list of strings representing the possible options.

    Returns:
    str or bool: The letter from the ASCII uppercase sequence corresponding to the matched option if found; otherwise, False.
    """
    in_option = [answer == option for option in options]
    if any(in_option):
        return ascii_uppercase[in_option.index(True)]
    else:
        return False

def get_valid_letters(options_lst, start=0):
    """
    Generates a list of valid ASCII uppercase letters that can be used to label options.

    The number of letters returned is based on the total number of options across all sublists in the input list.

    Parameters:
    options_lst (list of lists): A list of sublists, each containing strings of options.
    start (int): The starting index in the ASCII uppercase sequence from which to generate letters.

    Returns:
    list: A list of ASCII uppercase letters starting from the specified index, with the count equal to the total number of options.
    """
    total_length = sum(len(sublist) for sublist in options_lst)
    return list(ascii_uppercase[start:total_length])

# Check if the answer is in the options, either as a letter or full text returning the option letter
def check_answer(answer, options_lst, question_num):
    """
    Checks if the provided answer is valid for a given question, either as an option letter or as full text.

    It verifies if the answer is a correct option letter or matches the full text of the options. If the answer is valid,
    the corresponding option letter is returned. Otherwise, it returns a string indicating the answer is not in the options.

    Parameters:
    answer (str): The answer to check.
    options_lst (list of lists): A list of sublists, each containing strings of options for each question.
    question_num (int): The index of the question for which to check the answer.

    Returns:
    str: The option letter if the answer is valid; otherwise, a string indicating the answer is not in the options.
    """
    valid_letters = get_valid_letters(options_lst)
    if is_option(answer, options_lst[question_num]):
        return is_option(answer, options_lst[question_num])
    elif answer in valid_letters:
        return answer
    return 'ANSWER_NOT_IN_OPTION_TEXT'


def sanitize_text(text):
    """
    Sanitizes text to be used in a regex matching group.

    This function escapes special characters in the text that have special meaning in regular expressions,
    ensuring that the text can be safely used as a matching group.

    Parameters:
    text (str): The text to be sanitized.

    Returns:
    str: The sanitized text.
    """
    if type(text) != str:
        text = str(text)
    special_chars = ['\\', '.', '^', '$', '*', '+', '?', '{', '}', '[', ']', '(', ')', '|']
    sanitized_text = ''
    for char in text:
        if char in special_chars:
            sanitized_text += '\\' + char
        else:
            sanitized_text += char
    return sanitized_text

def parse_response(model_output, options_lst, question_num, return_val='PARSER_FOUND_NOTHING'):
    """
    Parses a model's output to find a valid answer within the specified options.

    The function looks for an answer in the model's output text that matches either the option letters or the full option texts.
    If a valid answer is found, it is returned; otherwise, a specified return value is returned.

    Parameters:
    model_output (str): The text output from the model.
    options_lst (list of lists): A list of sublists, each containing strings of options for each question.
    question_num (int): The index of the question to check the answer for.
    return_val (str): The default return value if no valid answer is found.

    Returns:
    tuple: The original model output and either the found answer or the default return value.
    """
    if not model_output:
        return model_output, return_val
    
    # Building patterns for both option letters and full option texts
    letters_pattern = '|'.join(get_valid_letters(options_lst))
    options_pattern = '|'.join([sanitize_text(option) for option in options_lst[question_num]])
    
    # Compiling the regular expression pattern to find the answer in the model output
    pattern = re.compile(
        rf'(?:Correct Answer|correct answer|Answer|the correct option is|The correct option is|the correct answer is|The correct answer is):?\s*({letters_pattern}|{options_pattern})(?:[\"\'\s\.]|$)',
        re.IGNORECASE
    )
    matched_option = pattern.search(model_output)
    if matched_option:
        answer = matched_option.group(1)
        return model_output, check_answer(answer, options_lst, question_num)
    
    return model_output, return_val


#########################################################################################
#########################################################################################
###                                                                                   ###
###                                                                                   ###
###                                                                                   ###
###                             Model Scoring Code                                    ###
###                                                                                   ###
###                                                                                   ###
###                                                                                   ###
#########################################################################################
#########################################################################################

def get_true_labels(question_id, answers_df):
    try:
        return answers_df.query('question_id == @question_id')['correct_answer'].tolist()
    except KeyError:
        return answers_df.query('question_id == @question_id')['correct'].tolist()
    
def get_correct(question_id, answers_df, permuted_answer):
    true_labels = get_true_labels(question_id, answers_df)
    try:
        return true_labels.index(1) == permuted_answer
    except ValueError:
        try:
            return true_labels.index(True) == permuted_answer
        except ValueError:
            return np.nan

def get_random_acc(question_id, answers_df):
    true_labels = get_true_labels(question_id, answers_df)
    return sum(true_labels) / len(true_labels)

def compute_ece(predicted_probs, true_labels, n_bins=10):
    """
    Computes the Expected Calibration Error (ECE) of a set of predictions.

    Parameters:
    - predicted_probs (list or np.ndarray): The predicted probabilities for the positive class.
    - true_labels (list or np.ndarray): The actual labels (0 or 1).
    - n_bins (int): The number of bins to use for grouping predicted probabilities.

    Returns:
    - float: The Expected Calibration Error (ECE).

    Example:
    >>> predicted_probs = [0.6, 0.1, 0.1, 0.2]
    >>> true_labels = [1, 0, 0, 0]
    >>> compute_ece(predicted_probs, true_labels, n_bins=2)
    """

    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower, bin_upper = bin_limits[i], bin_limits[i+1]
        # Indices of predictions in the current bin
        in_bin = np.where((predicted_probs > bin_lower) & (predicted_probs <= bin_upper))[0]
        if len(in_bin) == 0:
            continue
        
        # Average predicted probability in the bin
        bin_pred_prob = np.mean(predicted_probs[in_bin])
        # Actual accuracy in the bin
        bin_accuracy = np.mean(true_labels[in_bin])
        
        # Weight by the number of predictions in the bin
        bin_weight = len(in_bin) / len(predicted_probs)
        
        # Contribution of this bin to the ECE
        bin_error = np.abs(bin_accuracy - bin_pred_prob) * bin_weight
        
        ece += bin_error
    
    return ece

# def get_random_acc()