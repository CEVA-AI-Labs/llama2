"""
This module is used to evaluate the perplexity of the quantized Llama-2-7b model with different
quantization configurations.
"""
import torch
# from transformers import LlamaForCausalLM, LlamaTokenizer, OPTForCausalLM, AutoTokenizer
# from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from ceva_modeling_llama import LlamaForCausalLM, LlamaDecoderLayer
from transformers import LlamaTokenizer
from liteml.ailabs_liteml.retrainer import RetrainerConfig, RetrainerModel
from liteml.ailabs_shared.load_config import load_config
import csv
from utils import evaluate, get_calibration_loader

if __name__ == '__main__':
    model_dir = 'meta-llama/Llama-2-7b-hf'
    seq_len = 1024
    print('Initializing tokenizer')
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)

    # config_list contains the model that you want to compare. You can remove or add configurations to this list.
    config_list = [
        # 'float',
        # 'configs/W4A8/w4a8_per_channel_per_channel_matmul_A_dynamic_custom_bits.yaml',
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
        # 'configs/w8a8_per_channel_per_tensor_matmul_C_static.yaml',
        # 'configs/w8a8_per_tensor_per_channel_matmul_B_dynamic.yaml'

        # 'configs/Eli/config_1.yaml',
        # 'configs/Eli/config_2.yaml',
        # 'configs/Eli/config_3.yaml',
        # 'configs/Eli/config_4.yaml',
        # 'configs/Eli/w8a8_matmul8_B_sym.yaml',
        # 'configs/Eli/w8a8_matmul8_B_asym.yaml',
        # 'configs/Eli/w8a8_matmul16_B_sym.yaml',
        # 'configs/Eli/w8a8_matmul16_B_asym.yaml',
        # 'configs/Eli/w8a8_matmul8x16_B_asym.yaml',
        # 'configs/Eli/w8a8_matmul8_B_asym_lm_head_16.yaml',
        # 'configs/Eli/smoothquant_w8a8.yaml'

        # W8A8
        # 'configs/Eli/roni_w8a8ptok_sym_matmul_token_tensor.yaml',
        # 'configs/Eli/roni_w8a8ptok_asym_matmul_token_tensor.yaml',
        # 'configs/Eli/roni_w8a8ptok_sym_matmul_token_token.yaml',
        # 'configs/Eli/roni_w8a8ptok_asym_matmul_token_token.yaml',
        # 'configs/Eli/roni_w8ptoka8ptok_asym_matmul_token_token.yaml'
        # 'configs/Eli/roni_w8a8ptok_sym_matmul_token_no_head_tensor.yaml'  # The dynamic configuration

        # W8A8 static
        # 'configs/Eli/static_roni_w8a8_sym_matmul_no_head_tensor.yaml',
        # 'configs/Eli/static_roni_w8a8_sym_matmul_tensor.yaml',
        # 'configs/Eli/static_roni_w8a8_sym_matmul_token.yaml',
        # 'configs/Eli/static_roni_w8a8ptok_sym_matmul_token.yaml',
        # 'configs/Eli/static_roni_w8a8ptok_asym_matmul_token.yaml',
        'configs/Eli/static_roni_w8a8ptok_sym_matmul_token_no_head_tensor.yaml',  # The mixed dynamic and static conf
        # 'configs/Eli/static_roni_W8ptokA8_sym_matmul_token.yaml',
        # 'configs/Eli/static_roni_W8ptokA8_sym_matmul_token_no_head_tensor.yaml',
        # 'configs/Eli/static_smoothquant_roni_w8a8ptok_sym_matmul_token.yaml',

        # W4A8
        # 'configs/Eli/roni_w4a8ptok_sym_matmul_token_tensor.yaml',
        # 'configs/Eli/roni_w4a8ptok_asym_matmul_token_tensor.yaml',
        # 'configs/Eli/roni_w4a8ptok_sym_matmul_token_token.yaml',
        # 'configs/Eli/roni_w4a8ptok_asym_matmul_token_token.yaml',
        # 'configs/Eli/roni_w4ptoka8ptok_asym_matmul_token_token.yaml'

        # SmoothQuant W4A8
        # 'configs/Eli/smoothquant/roni_smoothquant_a0p1_w4ptoka8ptok_asym_matmul_token_token.yaml',
        # 'configs/Eli/smoothquant/roni_smoothquant_a0p2_w4ptoka8ptok_asym_matmul_token_token.yaml',
        # 'configs/Eli/smoothquant/roni_smoothquant_a0p3_w4ptoka8ptok_asym_matmul_token_token.yaml',
        # 'configs/Eli/smoothquant/roni_smoothquant_a0p4_w4ptoka8ptok_asym_matmul_token_token.yaml',
        # 'configs/Eli/smoothquant/roni_smoothquant_a0p5_w4ptoka8ptok_asym_matmul_token_token.yaml',
        # 'configs/Eli/smoothquant/roni_smoothquant_a0p6_w4ptoka8ptok_asym_matmul_token_token.yaml',
        # 'configs/Eli/smoothquant/roni_smoothquant_a0p7_w4ptoka8ptok_asym_matmul_token_token.yaml',
        # 'configs/Eli/smoothquant/roni_smoothquant_a0p8_w4ptoka8ptok_asym_matmul_token_token.yaml',
        # 'configs/Eli/smoothquant/roni_smoothquant_a0p85_w4ptoka8ptok_asym_matmul_token_token.yaml',


    ]
    ppl_list = []
    for config_name in config_list:
        print(config_name)
        print('Loading model')
        model = LlamaForCausalLM.from_pretrained(model_dir, device_map='auto', torch_dtype=torch.float16)
        # model = LlamaForCausalLM.from_pretrained(model_dir, device_map='auto', torch_dtype=torch.float32)
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

            # model._model._model.lm_head.set_weights_quant(False)
            # model._model._model.lm_head.set_data_quant(False)
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
