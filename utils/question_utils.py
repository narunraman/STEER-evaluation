from string import ascii_lowercase, ascii_uppercase
ALPHABET = list(ascii_uppercase)

import pandas as pd

from utils.utils import get_option_letters, flatten_list

SUFFIX_OPTIONS = {
    'mc':"\nAnswer by writing the option letter corresponding to the correct option. WRITE ONLY A SINGLE LETTER. \n\nCorrect Answer: ",
    'explain-answer': "\nBegin by explaining your reasoning in 2-3 sentences, enclosed in triple quotes. After your explanation, select and state the correct answer by writing 'Correct Answer: ' followed by your choice. BEGIN WITH YOUR EXPLANATION AND WRITE THE CORRECT ANSWER AT THE END",
    'explanation': "\nBriefly explain your reasoning in triple quotes.",
}


#########################################################################################
##                                                                                     ##
##                                                                                     ##
##                                                                                     ##
##                            Generate Question Code                                   ##
##                                                                                     ##
##                                                                                     ##
##                                                                                     ##
#########################################################################################


'''
input: 
- answer: str -- letter associated with the correct answer in the original ordering
- permutation: list -- list that was used to permute the options for inference
'''
def reverse_permutation_answer(answer, permutation):
    letters = list(ascii_lowercase)
    try:
        return letters[permutation[letters.index(answer.lower())]].upper()
    except IndexError:
        return answer


def reconstruct_context(prefix, questions, outputs, chat_type):
    """
    Reconstructs the context for different types of chat models based on the provided prefix, questions, and outputs.
    NOTE: this function only reconstructs the previous context and the subsequent question will need to be appended for the model to respond.

    Parameters:
    prefix (str): An initial text or conversation starter that is used as the beginning of the context.
    questions (list of str): A list of questions or user inputs that have been part of the conversation.
    outputs (list of str): A list of responses or outputs from the chat model corresponding to each question.
    chat_type (str): The type of chat model for which the context is being reconstructed. This determines the format
                     of the reconstructed context. Supported types include 'textual' for models like Falcon Instruct,
                     'list' for models like Llama2, ChatGPT, Anthropic, and a default format for non-chat models.

    Returns:
    str or list: The reconstructed context as a string or a list of dictionaries with 'role' and 'content' keys,
                 depending on the chat_type. For 'textual', it returns a single string with the conversation formatted
                 in a user and assistant turn-based manner. For 'list', it returns a list of messages, each represented
                 as a dictionary. For other types, it concatenates questions and outputs in order, separated by new lines.

    Example usage:
    context = reconstruct_context("Hello", ["How are you?"], ["I'm good, thanks!"], 'textual')
    """
    # For chat models like Falcon Instruct
    if chat_type == 'textual':
        context = prefix + '\n'
        for question, output in zip(questions, outputs):
            context += f"User: {question}\n"
            context += f"Falcon: {output}\n"
        return context
    # For chat models like Llama2, ChatGPT, Anthropic
    elif chat_type == 'list':
        if prefix:
            messages = [{'role': 'user', 'content': prefix}]
        else:
            messages = []
        for question, output in zip(questions, outputs):
            messages.append({'role': 'user', 'content': question})
            messages.append({'role': 'assistant', 'content': output})
        
        # Combine any consecutive user messages
        for i in range(1, len(messages)):
            if messages[i]['role'] == 'user' and messages[i-1]['role'] == 'user':
                messages[i]['content'] = messages[i-1]['content'] + '\n' + messages[i]['content']
                messages[i-1] = None
        messages = [message for message in messages if message is not None]

        return messages
    
    # For non-chat models
    return prefix + '\n' + '\n'.join([question + '\n' + output for output, question in zip(outputs, questions)])

def append_question(context, question, chat_type):
    if chat_type == 'list':
        if context and context[-1]['role'] == 'user':
            context[-1]['content'] += '\n' + question
        else:
            context.append({'role': 'user', 'content': question})
    elif chat_type == 'textual':
        user_roles = context.split('User: ')
        if 'Falcon: ' in user_roles[-1]:
            context += f"User: {question}\n"
        else:
            context += '\n' + question
    else:
        context += '\n' + question
    return context


def shuffle_and_permute_options(options):
    """
    Shuffle the options DataFrame conditioned on a question_id

    Args:
        options (pd.DataFrame): DataFrame containing options for a specific question.

    Returns:
        pd.DataFrame, list: the shuffled options DataFrame and a permutation
    """
    options_shuffled = options.sample(frac=1)  # Shuffle options
    permutation = options_shuffled.option_id.tolist()  # Get the permutation
    return options_shuffled, permutation

def reshape_alphabet(permutations):
    new_2d = []
    index = 0  # To keep track of the current position in the alphabet

    for permutation in permutations:
        # Take the next slice from one_d_list of length equal to the length of the current row
        new_row = ALPHABET[index:index + len(permutation)]
        new_2d.append(new_row)
        index += len(permutation)  # Move the index

    return new_2d

def permute_answer(model_answers, permutations):
    # reshape alphabet into options shape
    options_list = reshape_alphabet(permutations)
    # permute reshaped alphabet
    permuted_options = [[options[i] for i in permutation] for permutation, options in zip(permutations, options_list)]
    # get index of model output in permuted reshaped alphabet
    try:
        permuted_indices = [permuted_options[i].index(model_answer) for i, model_answer in enumerate(model_answers)]
    except ValueError:
        permuted_indices = [0] * len(model_answers)
        # print(f"Model answer not found in permuted options: {model_answers}")
        # print(f"Permutations: {permutations}")
        # sys.exit()
    return permuted_indices

def convert_probabilities(probabilities, q_id, permutations):
    
    # Need to pad out the probabilities dict to contain the full support
    # reshape alphabet into options shape
    options_list = reshape_alphabet(permutations)
    # permute reshaped alphabet
    permuted_options = [[options[i] for i in permutation] for permutation, options in zip(permutations, options_list)]
    curr_permuted_options = permuted_options[q_id]
    for option_letter in curr_permuted_options:
        if option_letter not in probabilities:
            probabilities[option_letter] = 0.0

    probabilities_list = [probabilities[key] for key in sorted(probabilities.keys())]
    return [probabilities_list[i] for i in permutations[q_id]]

def build_options_string(options, start_index):
    """
    Construct a string representing question options with global labels, and return updated global option index.

    Args:
        options (pd.DataFrame): DataFrame containing options for a specific question.
        alphabet (str): String containing the alphabet used for labeling options.
        start_index (int): Starting index for global option labeling.

    Returns:
        str, int: A string of options labeled with globally unique characters from the alphabet,
                  and the updated global option index after processing these options.
    """
    options_str = ""
    for i, option in enumerate(options.itertuples(), start=start_index):
        label = ALPHABET[i % len(ALPHABET)]
        options_str += f"{label}. {option.option_text}\n"
    updated_global_option_index = start_index + len(options)
    return options_str.strip(), updated_global_option_index

def format_question(question_text, options_str, params):
    """
    Format a question and its options based on specified parameters.

    Args:
        question_text (str): The text of the question.
        options_str (str): Formatted options string.
        params (dict): Parameters for formatting the question.

    Returns:
        list: A list containing formatted question strings.
    """
    question_type = params.get('question_type', 'mc')  # Default to 'mc' if not specified
    question_variants = []

    if question_type == 'mc' or question_type == 'mc-separate':
        suffix = SUFFIX_OPTIONS.get('mc')
        question_variants.append(f"{question_text}\n{options_str}\n{suffix}")
    elif question_type == 'explanation':
        suffix = SUFFIX_OPTIONS.get('explain-answer')
        question_variants.append(f"{question_text}\n{options_str}\n{suffix}")
    elif question_type == 'sequential-shown':
        suffix = SUFFIX_OPTIONS.get('explanation')
        question_variants.append(f"{question_text}\n{options_str}\n{suffix}")
        suffix = SUFFIX_OPTIONS.get('mc')
        question_variants.append(f"{options_str}\n{suffix}")
    elif question_type == 'sequential-hidden':
        suffix = SUFFIX_OPTIONS.get('explanation')
        question_variants.append(f"{question_text}\n{suffix}")
        suffix = SUFFIX_OPTIONS.get('mc')
        question_variants.append(f"{options_str}\n{suffix}")
    
    return question_variants

def add_answer(question, answer, params):
    question_type = params.get('question_type', 'mc')  # Default to 'mc' if not specified
    
    for i, part in enumerate(question):
        if question_type == 'mc' or question_type == 'mc-separate':
            question[i] += answer[i][1]
        elif question_type == 'explanation':
            answer_text = f"\n'''{answer[i][0]}'''\nCorrect Answer: {answer[i][1]}"
            question[i] += answer_text
        elif i % 2 == 0 and (question_type == 'sequential-shown' or question_type == 'sequential-hidden'):
            answer_text = f"\n'''{answer[i//2][0]}'''\n"
            question[i] += answer_text
        elif i % 2 == 1 and (question_type == 'sequential-shown' or question_type == 'sequential-hidden'):
            question[i] += answer[i-1][1]
    
    return question

def merge_dfs(questions_df, options_df, questions_metadata=None, answers_df=None):
    if questions_metadata is not None:
        if answers_df is not None:
            return options_df.merge(questions_df, on='question_id', how='left').merge(questions_metadata, on='question_id', how='left').merge(answers_df, on=['question_id', 'option_id'], how='left')
        return options_df.merge(questions_df, on='question_id', how='left').merge(questions_metadata, on='question_id', how='left')
    return pd.merge(options_df, questions_df, on='question_id', how='left')

def get_test_questions(base_id, questions_df, options_df, params):
    """
    Generate formatted test questions and their options based on specified parameters, including permutations of options.

    Args:
        base_id (str): Base question ID used to filter related questions.
        questions_df (pd.DataFrame): DataFrame containing question details.
        options_df (pd.DataFrame): DataFrame containing option details for questions.
        params (dict): Dictionary containing parameters for question formatting.
                       Expected keys are:
                       - 'question_type': Specifies the format of the questions and options.

    Returns:
        list, list, list: A tuple containing three lists:
                    - A list of formatted test questions including their options, modified based on 'question_type'.
                    - A list of formatted test options according to their permutation
                    - A list of permutations used for each question's options.
    
    Each question's options are permuted randomly, and the permutation is returned along with the formatted questions.
    """
    
    related_questions = questions_df[questions_df['question_id'].str.startswith(f"{base_id}_")]
    test_questions, test_options = [] ,[]
    options_permutations = []  # To track permutations of options for each question
    global_option_index = 0

    for _, question in related_questions.iterrows():
        question_id = question['question_id']
        question_text = question['question_text']
        options = options_df[options_df['question_id'] == question_id]
        
        # Shuffle options and generate permutation
        options, permutation = shuffle_and_permute_options(options)
        test_options.append(options.option_text.tolist())
        options_str, updated_global_option_index = build_options_string(options, global_option_index)
        
        # Update global option index
        global_option_index = updated_global_option_index

        # Store the permutation
        options_permutations.append(permutation)

        formatted_question = format_question(question_text, options_str, params)
        test_questions.extend(formatted_question)

    return test_questions, test_options, options_permutations

def build_prefix_string(base_id, questions_df, options_df, answers_df, params):
    
    prefix_questions, _, permutations = get_test_questions(base_id, questions_df, options_df, params)

    # flattened list of option letters permuted on each sub_id
    option_letters = flatten_list([[options[i] for i in permutation] for options, permutation in zip(get_option_letters(permutations), permutations)])
    # n-hot encoding of the correct answer indices
    try:
        correct_answer_indices = answers_df[answers_df['question_id'].str.startswith(f"{base_id}_")]['correct_answer'].tolist()
    except KeyError:
        correct_answer_indices = answers_df[answers_df['question_id'].str.startswith(f"{base_id}_")]['correct'].tolist()

    # Correct option letters
    correct_answers = [element for element, flag in zip(option_letters, correct_answer_indices) if flag == 1]

    # get explanations
    explanations = questions_df[questions_df['question_id'].str.startswith(f"{base_id}_")]['explanation'].tolist()
    answers = [(explanations[i], correct_answers[i]) for i in range(len(correct_answers))]
    
    # add answers and/or explanations to the prefix
    prefix_questions = add_answer(prefix_questions, answers, params)
    
    # convert list to string
    prefix_string = '\n'.join(prefix_questions)
    return prefix_string


def build_prefix(task_data, questions_df, options_df, questions_metadata, answers_df, params):

    # get num_shots number of prefix questions by question_id
    prefix_questions = questions_df.query('explanation != False').merge(questions_metadata, on='question_id', how='left').query("type == @task_data['type'] and domain == @task_data['domain'] and difficulty_level == @task_data['difficulty_level']")
    try:
        prefix_question_ids = prefix_questions.sample(n=params['num_shots'])['question_id'].tolist()
    except ValueError as e:
        if 'Cannot take a larger sample than population' in str(e):
            print(f"Tried sampling {params['num_shots']} from dataframe with metadata {task_data}. Only {len(prefix_questions)} available.")
        raise e
    # Get base ids
    base_ids = set(question_id.split('_')[0] for question_id in prefix_question_ids)

    return '\n\n'.join([build_prefix_string(base_id, questions_df, options_df, answers_df, params) for base_id in base_ids])