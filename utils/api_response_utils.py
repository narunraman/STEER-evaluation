
import openai



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


def get_mc(client, valid_tokens, context, model, use_logprobs, num_logprobs):
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
    'mc': get_mc,
}