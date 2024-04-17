import re
from string import ascii_lowercase, ascii_uppercase


OPTIONS = list(ascii_uppercase)
LETTERS = list(ascii_lowercase)

#########################################################################################
#########################################################################################
###                                                                                   ###
###                                                                                   ###
###                                                                                   ###
###                             Parse Results Code                                    ###
###                                                                                   ###
###                                                                                   ###
###                                                                                   ###
#########################################################################################
#########################################################################################

def find_answer_letter(text):
    # This regex looks for the phrase "Correct Answer:" followed by a single space and a single letter (A-Z, a-z)
    # at the end of the string. The '$' asserts the position at the end of the string.
    match = re.search(r'Correct Answer: ([A-Za-z])$', text.rstrip(), re.IGNORECASE)
    if match:
        return match.group(1)  # Return the letter found
    else:
        return None  # Return None if no match is found

def remove_answer_letter(text):
    # This regex looks for "Correct Answer: " followed by a single letter at the end of the string
    # and replaces just the letter with an empty string, effectively removing it.
    modified_text = re.sub(r'(?<=Correct Answer: )[A-Za-z]$', '', text.rstrip(), flags=re.IGNORECASE)
    return modified_text




#########################################################################################
#########################################################################################
###                                                                                   ###
###                                                                                   ###
###                                                                                   ###
###                           Restrict Output Code                                    ###
###                                                                                   ###
###                                                                                   ###
###                                                                                   ###
#########################################################################################
#########################################################################################


# NOTE: not sure if restricting the output is a fair evaluation of an LLM.
def restrict_phrases(input_str, tokenizer, num_options):
    """Restricts the answer to a fixed set of allowed phrases"""
    # TODO: This implementation is inefficient, but there's plenty of obvious ways to speed it up
    # https://discuss.huggingface.co/t/example-of-prefix-allowed-tokens-fn-while-text-generation/6635/2
    def prefix_allowed_tokens(batchId, inputIds):
        # Get the answer so far
        decoded = tokenizer.decode(inputIds)[len(input_str):]
        # How could we continue this into a phrase
        phrases = [p[len(decoded):] for p in LETTERS[:num_options] if p.startswith(decoded)]
        # What token comes next?
        next_tokens = set(tokenizer.encode(p)[0] if p else tokenizer.eos_token_id for p in phrases)
        return list(next_tokens)
    return prefix_allowed_tokens


# NOTE: not sure if restricting the output is a fair evaluation of an LLM.
def restrict_letters(input_str, tokenizer, num_options):
    """Only allows answers to start with A or B or .."""
    input_len = len(tokenizer.encode(input_str))
    vocab = tokenizer.get_vocab()
    all_inputs = list(vocab.values())
    
    def normalize(t):
        """Normalize tokens"""
        return t.replace("Ġ", "").lower()

    def get_inputs(f):
        """Gets all input ids with tokens matching a function"""
        return [v for (k,v) in vocab.items() if f(k)]
    
    letter_inputs = get_inputs(lambda t: normalize(t) in LETTERS[:num_options])
    def prefix_allowed_tokens(batchId, inputIds):
        if len(inputIds) > input_len:
            return all_inputs
        return letter_inputs
    
    return prefix_allowed_tokens
