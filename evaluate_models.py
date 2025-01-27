"""
This module is used to evaluate the perplexity of the quantized Llama-2-7b model with different
quantization configurations.
"""
import torch
# from transformers import LlamaForCausalLM, LlamaTokenizer, OPTForCausalLM, AutoTokenizer
# from transformers.models.llama.modeling_llama import LlamaDecoderLayer
# from ceva_modeling_llama import LlamaForCausalLM, LlamaDecoderLayer, LlamaRMSNorm  # v4_34
from ceva_modeling_llama_v4_46 import LlamaForCausalLM, LlamaDecoderLayer, LlamaRMSNorm  # v4_46
from transformers import LlamaTokenizer
from liteml.ailabs_liteml.retrainer import RetrainerConfig, RetrainerModel
from liteml.ailabs_shared.load_config import load_config
import csv
from utils import evaluate, get_calibration_loader


def load_spinquant_weights(model, spinquant_model_path):
    """
    This function wraps Llama model with spinquant weights.
    """
    orig_state_dict = model.state_dict()
    spinquant_state_dict = torch.load(spinquant_model_path)
    spinquant_float_state_dict = {key: spinquant_state_dict[key] for key in orig_state_dict}
    model.load_state_dict(spinquant_float_state_dict)


if __name__ == '__main__':
    model_dir = 'meta-llama/Llama-2-7b-hf'
    seq_len = 2048
    enable_spinquant = True
    print('Initializing tokenizer')
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)

    # config_list contains the model that you want to compare. You can remove or add configurations to this list.
    config_list = [
        # 'float',
        # 'configs/w8a8_per_tensor_per_token_dynamic.yaml',  # The dynamic configuration
        # 'configs/w8a8_static.yaml',  # the static configuration
        # 'configs/w8a8_npm_v1_3_4.yaml',  # The mixed dynamic and static configuration
        #'configs/spinquant/w4a8_spinquant_e.yaml',
        'configs/spinquant/w4a8_spinquant_e_softmax.yaml',
        #'configs/spinquant/w4a8_spinquant_e_matmul.yaml',
        #'configs/spinquant/w4a8_spinquant_e_Silu.yaml',
        #'configs/spinquant/w4a8_spinquant_e_all.yaml',
        #'configs/spinquant/w4a8_spinquant_e_rmsnorm.yaml',
        #'configs/spinquant/w4a8_spinquant_e_no_rmsnorm.yaml',
        #'configs/Rana/fack_quant_RMSnorm.yaml',

    ]

    #spinquant_path = "/projects/systems/systems/Ranam/SpinQuant/saved_models/spinquant_gptq_group128.pth"
    spinquant_path = "spinquant_gptq_group128.pth"

    ppl_list = []
    for config_name in config_list:
        print(config_name)
        print('Loading model')
        model = LlamaForCausalLM.from_pretrained(model_dir, device_map='cuda:0', torch_dtype=torch.float16, attn_implementation="eager")
        # model = LlamaForCausalLM.from_pretrained(model_dir, device_map='auto', torch_dtype=torch.float32)

        # wrap model with spinquant
        if enable_spinquant:
            load_spinquant_weights(model, spinquant_path)

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


            #if "fack_quant_RMSnorm" in config_name:
            if "w4a8_spinquant_e_rmsnorm" in config_name or "w4a8_spinquant_e_all" in config_name:
                layers = model._model._model.model.layers
                for layer in layers:
                    ### if the min input in the layer is smaller than value x (0.03125 give best results), choose alpha, otherwise alpha=1
                    layernorm = layer.input_layernorm
                    min_layernorm = layer.input_layernorm._quantizer_mean.obs.min_val
                    if min_layernorm < 0.03125:
                        alpha_layernorm = 0.03125 / min_layernorm
                        layer.input_layernorm._alpha = alpha_layernorm

                    post_attn = layer.post_attention_layernorm
                    min_post_attn = layer.post_attention_layernorm._quantizer_mean.obs.min_val
                    if min_post_attn < 0.03125:
                        alpha_post_attn = 0.03125 / min_post_attn
                        layer.post_attention_layernorm._alpha = alpha_post_attn

            ## model._model._model.lm_head.set_weights_quant(False)
            ## model._model._model.lm_head.set_data_quant(False)
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
