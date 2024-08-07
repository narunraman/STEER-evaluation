
# External packages
import torch
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


# def get_random_acc()