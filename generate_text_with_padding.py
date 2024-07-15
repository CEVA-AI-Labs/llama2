"""
This module demonstrates text generation with the quantized Llama2-7b-chat model.
"""
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, OPTForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import time
from liteml.ailabs_liteml.retrainer import RetrainerConfig, RetrainerModel
from liteml.ailabs_shared.load_config import load_config
from utils import get_calibration_loader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_text(prompt, model, tokenizer):
    print('Initializing pipeline')
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
    )

    print('Running llama')
    time1 = time.time()
    sequences = pipeline(
        text_inputs=prompt,
        do_sample=True,
        top_k=1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        # truncation=True,
        max_length=400,
    )
    end = time.time()
    print(f'Generating text took {end - time1}:.2f seconds')
    for i, seq in enumerate(sequences):
        print(f'Result {i}:')
        print(f"{seq['generated_text']}")


if __name__ == '__main__':
    # model_dir = 'meta-llama/Llama-2-7b-hf'
    model_dir = 'meta-llama/Llama-2-7b-chat-hf'
    # prompt = 'The meaning of life is'
    # prompt = ['How are you', 'The meaning of life is']
    # prompt = ["A list of colors: red, blue", "Portugal is"]
    prompt = ["1, 2, 3", "A, B, C, D, E, F"]

    print('Initializing tokenizer')
    tokenizer = LlamaTokenizer.from_pretrained(model_dir, padding_side='right', padding=True)
    # tokenizer.pad_token = tokenizer.eos_token

    # tokenizer.pad_token_id = 0
    # Select the configuration you wish to test
    config_name = 'float'


    print('Loading model')
    model = LlamaForCausalLM.from_pretrained(model_dir, device_map='auto', torch_dtype=torch.float16)

    if '<pad>' not in tokenizer.get_vocab():
        # Add the pad token
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # Resize the embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Configure the pad token in the model
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    # Check if they are equal
    assert model.config.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"

    tokenized_input = tokenizer(prompt, return_tensors="pt", padding=True)

    # print("Tokenized Text:", [tokenizer.decode([x]) for x in tokenized_input["input_ids"][0]])
    # print("Token IDs:", tokenized_input["input_ids"][0])

    model.eval()
    with torch.no_grad():
        # generate_text(prompt, model, tokenizer)  # for float model
        # generate_ids = model.generate(tokenized_input.input_ids, max_length=100)  # don't mask padded tokens
        generate_ids = model.generate(tokenized_input.input_ids, attention_mask=tokenized_input.attention_mask, max_length=100)  # mask padded tokens
        out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(out[0])
