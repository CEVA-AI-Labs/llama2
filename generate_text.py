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
    prompt = 'The meaning of life is'

    print('Initializing tokenizer')
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)

    # Select the configuration you wish to test
    # config_name = 'float'
    # config_name = 'configs/smoothquant_w8a8_per_channel_per_channel_dynamic.yaml'
    # config_name = 'configs/smoothquant_w8a8_per_channel_per_tensor_matmul_dynamic.yaml'
    # config_name = 'configs/smoothquant_w8a8_per_channel_per_tensor_matmul_static.yaml'
    # config_name = 'configs/w4a8_per_channel_per_channel_dynamic.yaml'
    # config_name = 'configs/w4a8_per_channel_per_tensor_matmul_dynamic.yaml'
    # config_name = 'configs/w8a8_per_channel_per_channel_matmul_A_dynamic.yaml'
    # config_name = 'configs/w8a8_per_channel_per_tensor_matmul_dynamic.yaml'
    # config_name = 'configs/w8a8_per_channel_per_tensor_matmul_static.yaml'
    # config_name = 'configs/w8a8_per_tensor_per_channel_dynamic.yaml'  # ?
    # config_name = 'configs/Eli/roni_w8a8ptok_sym_matmul_token_tensor.yaml'
    # config_name = 'configs/Eli/roni_w8ptoka8ptok_asym_matmul_token_token.yaml'
    config_name = 'configs/Eli/static_roni_w8a8ptok_sym_matmul_token_no_head_tensor.yaml'

    # config_name = 'test_config_llama.yaml'

    print('Loading model')
    model = LlamaForCausalLM.from_pretrained(model_dir, device_map='auto', torch_dtype=torch.float16)
    # model = LlamaForCausalLM.from_pretrained(model_dir, device_map=0, torch_dtype=torch.float16)
    model.eval()
    with torch.no_grad():
        if config_name != 'float':
            conf = load_config(config_name)
            if 'SmoothQuant' in conf:
                conf["SmoothQuant"]["decoder_class"] = LlamaDecoderLayer
            if conf['QAT']['data_quantization']['quantization_mode'] == 'static':
                # Add calibration loader
                calib_loader = get_calibration_loader(tokenizer)
                conf["QAT"]["data_quantization"][
                    "calibration_loader"
                ] = calib_loader
                conf["QAT"]["data_quantization"][
                    "calibration_loader_key"
                ] = lambda model, x: model(x.cuda())
            model = RetrainerModel(model, config=RetrainerConfig(conf))
            if 'OmniQuant' in conf["QAT"]:
                model = model.to(device)

            generate_text(prompt, model._model._model, tokenizer)
        else:
            generate_text(prompt, model, tokenizer)  # for float model
