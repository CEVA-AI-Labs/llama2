"""
This module is used to evaluate the perplexity of the quantized Llama-2-7b model with different
quantization configurations.
"""
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, OPTForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from liteml.ailabs_liteml.retrainer import RetrainerConfig, RetrainerModel
from liteml.ailabs_shared.load_config import load_config
import csv
from utils import evaluate, get_calibration_loader

if __name__ == '__main__':
    model_dir = 'meta-llama/Llama-2-7b-hf'
    seq_len = 2048
    print('Initializing tokenizer')
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)

    # config_list contains the model that you want to compare. You can remove or add configurations to this list.
    config_list = [
        # 'float',
        # 'configs/W4A8/w4a8_per_channel_per_channel_matmul_A_dynamic.yaml',
        # 'configs/W4A8/w4a8_per_channel_per_channel_matmul_A_dynamic.yaml',
        # 'configs/W4A8/w4a8_per_channel_per_tensor_matmul_B_dynamic.yaml',
        # 'configs/W4A8/w4a8_per_channel_per_tensor_matmul_C_static.yaml',
        # 'configs/W4A8/smoothquant_w4a8_per_channel_per_channel_matmul_A_dynamic.yaml',
        # 'configs/W4A8/smoothquant_w4a8_per_channel_per_tensor_matmul_B_dynamic.yaml',
        # 'configs/W4A8/smoothquant_w4a8_per_channel_per_tensor_matmul_C_static.yaml',
        # 'configs/W4A8/smoothquant_a0p4_w4a8_per_channel_per_channel_matmul_A_dynamic.yaml',

        # 'configs/w4a8_per_channel_per_channel_matmul_A_dynamic.yaml',
        # 'configs/w4a8_per_channel_per_tensor_matmul_B_dynamic.yaml',
        # 'configs/smoothquant_w8a8_per_channel_per_channel_matmul_A_dynamic.yaml',
        # 'configs/smoothquant_w8a8_per_channel_per_tensor_matmul_B_dynamic.yaml',
        # 'configs/smoothquant_w8a8_per_channel_per_tensor_matmul_C_static.yaml',
        # 'configs/w8a8_per_channel_per_channel_matmul_A_dynamic.yaml',
        # 'configs/w8a8_per_channel_per_tensor_matmul_B_dynamic.yaml',
        'configs/w8a8_per_channel_per_tensor_matmul_C_static.yaml',
        # 'configs/w8a8_per_tensor_per_channel_matmul_B_dynamic.yaml'
    ]
    ppl_list = []
    for config_name in config_list:
        print(config_name)
        print('Loading model')
        model = LlamaForCausalLM.from_pretrained(model_dir, device_map='auto', torch_dtype=torch.float16)
        with torch.no_grad():
            if config_name != 'float':
                conf = load_config(config_name)
                if 'SmoothQuant' in conf:
                    conf["SmoothQuant"]["decoder_class"] = LlamaDecoderLayer
                if conf['QAT']['data_quantization']['quantization_mode'] == 'static':
                    # Add calibration loader
                    calib_loader = get_calibration_loader(tokenizer, seq_len=seq_len)
                    conf["QAT"]["data_quantization"][
                        "calibration_loader"
                    ] = calib_loader
                    conf["QAT"]["data_quantization"][
                        "calibration_loader_key"
                    ] = lambda model, x: model(x.cuda())
                model = RetrainerModel(model, config=RetrainerConfig(conf))

            ppl = evaluate(model, tokenizer, seq_len=seq_len)
            print(f'Model {config_name}')
            print(f'Perplexity: {ppl:.4}')
            ppl_list.append({'name': config_name, 'perplexity': ppl.item()})

            del model

    # Save results in a csv file
    fields = ['name', 'perplexity']
    with open('ppl.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ppl_list)
