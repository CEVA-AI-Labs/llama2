"""
This script is used to create chat application compatible with transformers version 4.46.
It doesn't apply padding at all.
It uses DynamicCache to store the chat history context.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import LlamaTokenizer
from ceva_modeling_llama_v4_46 import LlamaForCausalLM, LlamaDecoderLayer
import torch
from liteml.ailabs_liteml.retrainer import RetrainerConfig, RetrainerModel
from liteml.ailabs_shared.load_config import load_config
from utils import get_calibration_loader
from transformers.cache_utils import DynamicCache
import argparse
from typing import List, Union
import numpy as np
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    # Argument parser for IP address
    parser = argparse.ArgumentParser(description="FastAPI backend for Llama-2 chat application")
    parser.add_argument("--ip", type=str, required=True, help="IP address to run the FastAPI server on")
    args = parser.parse_args()
    return args


# Define the input schema
class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float
    max_tokens: int
    stop: Union[str, list]


set_seed(42)

# Initialize the FastAPI app
app = FastAPI()

# Load the Llama-2-7b-hf model and tokenizer
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
config_name = 'float'
# config_name = '../configs/w8a8_per_tensor_per_token_dynamic.yaml' # The dynamic quantization conf
# config_name = '../configs/w8a8_static.yaml'  # the static quantization conf
# config_name = '../configs/w8a8_npm_v1_3_4.yaml' # The mixed dynamic and static conf

try:
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"  # Automatically map the model to available GPUs/CPU
    )

    # Quantize model
    if config_name != 'float':
        conf = load_config(config_name)
        if 'SmoothQuant' in conf:
            conf["SmoothQuant"]["decoder_class"] = LlamaDecoderLayer
        if conf['QAT']['data_quantization']['quantization_mode'] == 'static':
            # Add calibration loader
            calib_loader = get_calibration_loader(tokenizer, seq_len=1024)
            conf["QAT"]["data_quantization"][
                "calibration_loader"
            ] = calib_loader
            conf["QAT"]["data_quantization"][
                "calibration_loader_key"
            ] = lambda model, x: model(x.cuda())
        with torch.no_grad():
            model = RetrainerModel(model, config=RetrainerConfig(conf))
        if 'OmniQuant' in conf["QAT"]:
            model = model.to(device)

except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load the model and tokenizer.")


past_key_values = DynamicCache()
max_cache_length = past_key_values.get_max_length()


@app.get("/")
def read_root():
    return {"message": "Welcome to the Llama-2-7b-hf Chat API"}


@app.post("/chat")
def chat(request: ChatRequest):
    """
    Generate a chat response based on the user's message and session history.
    """
    try:
        messages = request.messages
        inputs = tokenizer.apply_chat_template(messages,
                                               add_generation_prompt=True,
                                               return_tensors="pt",
                                               return_dict=True).to(model.device)
        input_length = inputs["input_ids"].shape[1]
        outputs = model.generate(**inputs, do_sample=True, max_length=4096)
        completion = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
        return {"model": "llama2_liteml", "choices": [{"message": {"role": "assistant", "content": completion}}]}


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during chat: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    args = parse_args()
    uvicorn.run(app, host=args.ip, port=8000)
