# Built-in packages
import os
from typing import Optional
from collections import defaultdict
from math import exp
import traceback
# External packages
from openai import OpenAI, AzureOpenAI, AsyncAzureOpenAI
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
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
        model: str = 'gpt-4-1106-preview',
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
            top_logprobs 
        )
        if logprobs:
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
        else:
            return response.choices[0].message.content

class AsyncGPTClient:
    def __init__(self, client='azure'):
        if client.lower() == 'azure':
            self.client = AsyncAzureOpenAI(
                api_key=os.environ['AZURE_OPENAI_API_KEY'],  
                api_version="2024-02-01",
                azure_endpoint = os.environ['AZURE_OPENAI_ENDPOINT']
            )
        elif client.lower() == 'openai':
            self.client = OpenAI(
                api_key=os.environ['OPENAI_API_KEY']
            )
    
    async def get_completion(
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

        completion = await self.client.chat.completions.create(**params)
        return completion

    async def get_explanation(
        self,
        messages: list[dict[str, str]],
        model: str = 'gpt-4-1106-preview',
        max_tokens = 500,
        temperature = 0,
        stop = None,
        seed = 123, 
        tools = None,
        logprobs = None,
        top_logprobs=None,
    ) -> str:
        response = await self.get_completion(
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
    
    async def get_answer(
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
            
            response = await self.get_completion(
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
            if logprobs:
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
            else:
                return response.choices[0].message.content
            

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

MODEL_PATH = '/home/narunram/scratch/models/'

def set_model_path(path: str):
    global MODEL_PATH
    MODEL_PATH = path

def get_flash_attention_models(model_name: str):
    if 'llama' in model_name.lower() or 'falcon' in model_name.lower() or 'mistral' in model_name.lower() or 'mixtral' in model_name.lower() or 'jamba' in model_name.lower() or 'dbrx' in model_name.lower():
        return True
    return False

def supports_flash_attention(device_id):
    """Check if a GPU supports FlashAttention."""
    major, minor = torch.cuda.get_device_capability(device_id)
    
    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0

    return is_sm8x or is_sm90

def load_model(base_path: str, from_pretrained_kwargs: dict):
    if not os.path.exists(base_path):
        print(f'Skipping model {base_path}. Cannot be found in {base_path}.')
        return False, False
    
    if 'alpaca' in base_path:
        tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast = False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    try:
        if "falcon" in base_path.lower():
            model = AutoModelForCausalLM.from_pretrained(base_path, **from_pretrained_kwargs)    
        else:
            model = AutoModelForCausalLM.from_pretrained(base_path, **from_pretrained_kwargs, trust_remote_code=True)
    except NameError:
        model = AutoModel.from_pretrained(base_path, low_cpu_mem_usage=True, **from_pretrained_kwargs, local_files_only=True, trust_remote_code=True)
    return model, tokenizer


def build_kwargs(device: str, model_name: str, num_gpus: int, max_gpu_mem: Optional[str] = None):
    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
    elif device == "cuda":
        gpu_supports_flash_attn = all([supports_flash_attention(i) for i in range(num_gpus)])
        if get_flash_attention_models(model_name) and gpu_supports_flash_attn:
            kwargs = {"torch_dtype": torch.bfloat16}
            kwargs['attn_implementation'] = "flash_attention_2"
        else:
            kwargs = {"torch_dtype": torch.float16}
        # if 'alpaca' in model_name:
            # kwargs['use_fast'] = False
        kwargs["local_files_only"] = True
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

def load_model_tokenizer(model_path: str, model_name: str, device: str = "cuda", num_gpus: int = 2, max_gpu_mem: Optional[str] = None):
    
    kwargs = build_kwargs(device, model_name, num_gpus, max_gpu_mem)
    
    model_path = os.path.join(model_path, model_name, 'snapshots/')
    model_path = os.path.join(model_path, os.listdir(model_path)[0])
    print('Loading model from:', model_path) 
    try:
        model, tokenizer = load_model(model_path, kwargs)
    except Exception as error:
        print(f'Error loading model: {error}')
        print(traceback.format_exc())
        return False, False


    if (device == "cuda" and num_gpus == 1) and model != False:
        model.to(device)

    return model, tokenizer