# Built-in packages
import os
from typing import Optional
from collections import defaultdict
from math import exp
# External packages
from openai import OpenAI, AzureOpenAI
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from accelerate import Accelerator
# Local packages
from utils.utils import get_gpu_memory, normalize_dict

class GPTClient:
    def __init__(self, client='azure'):
        if client.lower() == 'azure':
            self.client = AzureOpenAI(
                api_key=os.environ['AZURE_OPENAI_API_KEY'],  
                api_version="2024-02-01",
                azure_endpoint = os.environ['AZURE_OPENAI_ENDPOINT']
            )
        elif client.lower() == 'openai':
            self.client = OpenAI(
                api_key=os.environ['OPENAI_API_KEY']
            )
    
    def get_completion(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt-4-1106-preview",
        max_tokens=500,
        temperature=0,
        stop=None,
        seed=123,
        tools=None,
        logprobs=None,  # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message..
        top_logprobs=None,
    ) -> str:
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop,
            "seed": seed,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
        }
        if tools:
            params["tools"] = tools

        completion = self.client.chat.completions.create(**params)
        return completion

    def get_explanation(
        self,
        messages: list[dict[str, str]],
        model: str = 'gpt-4',
        max_tokens = 500,
        temperature = 0,
        stop = None,
        seed = 123, 
        tools = None,
        logprobs = None,
        top_logprobs=None,
    ) -> str:
        response = self.get_completion(
            messages,
            model,
            max_tokens,
            temperature,
            stop,
            seed,
            tools,
            logprobs,
            top_logprobs
        )
        return response.choices[0].message.content
    
    def get_answer(
        self,
        valid_tokens,
        messages: list[dict[str, str]],
        model: str = 'gpt-4',
        max_tokens = 500,
        temperature = 0,
        stop = None,
        seed = 123, 
        tools = None,
        logprobs = None,
        top_logprobs=None,
    ) -> dict:
        response = self.get_completion(
            messages,
            model,
            max_tokens,
            temperature,
            stop,
            seed,
            tools,
            logprobs,
            min(5, top_logprobs) # Azure currently only supports top 5 logprobs
        )
        top_responses = response.choices[0].logprobs.content[0].top_logprobs
        output = defaultdict(lambda: 0)
        for logprob in top_responses:
            for valid_token in valid_tokens:
                if valid_token.startswith(logprob.token.upper()):
                    output[logprob.token] = exp(logprob.logprob)*100
            if len(output) == len(valid_tokens):
                break
        output = normalize_dict(output)
        return max(output, key=output.get), output 


    

#########################################################################################
#########################################################################################
##                                                                                     ##
##                                                                                     ##
##                                                                                     ##
##                               Model Helper Code                                     ##
##                                                                                     ##
##                                                                                     ##
##                                                                                     ##
#########################################################################################
#########################################################################################



def load_model(device, args, model_name):
    # Check if model is downloaded
    if not os.path.exists(args['model_path']+model_name):
        print(f'Skipping model {model_name}. Cannot be found in {args["model_path"]}.')
        return False, False

    # loads model and tokenizer (both need to be in the same snapshot)
    try:
        model, tokenizer = load_model_tokenizer(args['model_path']+model_name+'/snapshots/', device=device, **args['models'][model_name])
        return model, tokenizer
    except Exception as error:
        print(f'Skipping {model_name.split("/")[-1]}, because: {error}')
        return False, False


# NOTE: make sure this points to scratch directory
def load_model(base_path: str, from_pretrained_kwargs: dict):
    # Initialize the accelerator
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(base_path, **from_pretrained_kwargs, local_files_only=True, token=os.environ['HF_TOKEN'])
        # Prepare the model for distributed inference using `device_map='auto'`
        model = accelerator.prepare(model, device_map='auto')
    except NameError:
        model = AutoModel.from_pretrained(base_path, low_cpu_mem_usage=True, **from_pretrained_kwargs, local_files_only=True, token=os.environ['HF_TOKEN'])
    return model, tokenizer


def build_kwargs(device: str, num_gpus: int, max_gpu_mem: Optional[str] = None):
    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus != 1:
            kwargs["device_map"] = "auto"
            if max_gpu_mem is None:
                kwargs["device_map"] = "sequential"  # This is important for not the same VRAM sizes
                available_gpu_memory = get_gpu_memory(num_gpus)
                kwargs["max_memory"] = {i: str(int(available_gpu_memory[i] * 0.85)) + "GiB" for i in range(num_gpus)}
            else:
                kwargs["max_memory"] = {i: max_gpu_mem for i in range(num_gpus)}
    else:
        raise ValueError(f"Invalid device: {device}")

    return kwargs

def load_model_tokenizer(model_path: str, device: str = "cuda", num_gpus: int = 2, max_gpu_mem: Optional[str] = None):
    """Load a model from Hugging Face."""

    kwargs = build_kwargs(device, num_gpus, max_gpu_mem)
    model_path += os.listdir(model_path)[0] 
    model, tokenizer = load_model(model_path, kwargs)

    if (device == "cuda" and num_gpus == 1):
        model.to(device)

    return model, tokenizer