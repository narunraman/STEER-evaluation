# Built-in packages
from string import ascii_uppercase
import re
from collections import defaultdict

# External packages
import numpy as np


#########################################################################################
#########################################################################################
###                                                                                   ###
###                                                                                   ###
###                                                                                   ###
###                                Adaptation Code                                    ###
###                                                                                   ###
###                                                                                   ###
###                                                                                   ###
#########################################################################################
#########################################################################################




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


