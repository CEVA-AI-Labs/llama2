import torch
import torch.nn as nn
import transformers
# from transformers import LlamaForCausalLM, LlamaTokenizer
# from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, _expand_mask, _make_causal_mask
from transformers import LlamaTokenizer, AutoTokenizer
from ceva_modeling_llama import LlamaForCausalLM, LlamaDecoderLayer, LlamaRMSNorm, _expand_mask, _make_causal_mask

from huggingface_hub import login
import time

def replace_norm(model):
    for name, m in model.model.named_modules():
        if isinstance(m, LlamaDecoderLayer):
            print(f'Replacing LlamaRMSNorm of layer {name}.')
            m.input_layernorm = nn.LayerNorm(len(m.input_layernorm.weight), eps=1e-6)
            m.post_attention_layernorm = nn.LayerNorm(len(m.post_attention_layernorm.weight), eps=1e-6)


def prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


def create_inputs(tokens, stage, dtype):
    if stage == 'prompt':
        # Prompt stage
        input_embeds = torch.rand((1, tokens, 4096), dtype=dtype)
        attention_mask = torch.ones((1, tokens), dtype=dtype, device=device)
        attention_mask = prepare_decoder_attention_mask(attention_mask, input_shape=(1, tokens), inputs_embeds=input_embeds, past_key_values_length=0)
        position_ids = torch.arange(0, tokens, device=device).unsqueeze(dim=0)
        past_key_value = None
        use_cache = True

    else:
        # Decode stage
        input_embeds = torch.rand((1, 1, 4096), dtype=dtype)
        # KV cache. For decode stage use x in the shape of (1, 1, 4096)
        past_key_value = (torch.randn((1, 32, tokens, 128), dtype=dtype, device=device),
                                 torch.randn((1, 32, tokens, 128), dtype=dtype, device=device))
        position_ids = torch.tensor(((tokens,),), device=device)
        attention_mask = torch.ones((1, tokens + 1), dtype=dtype, device=device)
        attention_mask = prepare_decoder_attention_mask(attention_mask, input_shape=(1, 1), inputs_embeds=input_embeds, past_key_values_length=tokens)
        use_cache = True

    model_inputs = {'hidden_states': input_embeds,
                    'past_key_value': past_key_value,
                    'position_ids': position_ids,
                    'attention_mask': attention_mask,
                    'use_cache': use_cache}
    return model_inputs


model_dir = 'meta-llama/Llama-2-7b-hf'
print('Loading model')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stage = 'prompt'  # 'prompt' or 'decode'
# model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', device_map='auto', torch_dtype=torch.float16)
model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', device_map='auto', torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
model.eval()

# Input to the model
tokens = 1024
model_inputs = create_inputs(tokens, stage, dtype=torch.float32)
# model_inputs = torch.rand((1, tokens, 4096), dtype=torch.float16)

# replace_norm(model)
# model.to(device)
# model = model.half()

print('Modified model:')
print(model)
for name, module in model.named_modules():
    print(name)
    if isinstance(module, LlamaDecoderLayer) or name == 'model.norm' or name == 'lm_head':
        torch.onnx.export(module,  # model being run
                            model_inputs,  # model input (or a tuple for multiple inputs)
                            # f'/projects/vbu_projects/users/royj/LinuxProjects/onnx_model/Llama2DecoderFP16_tokens_{tokens}/{name}_{stage}.onnx',  # where to save the model (can be a file or file-like object)
                            # f'/projects/vbu_projects/users/royj/LinuxProjects/onnx_model/Llama2DecoderFP16_debug/{name}_{stage}.onnx',  # where to save the model (can be a file or file-like object)
                            f'/projects/vbu_projects/users/royj/LinuxProjects/onnx_model/Llama2DecoderFP32/{name}_{stage}_v2.onnx',  # where to save the model (can be a file or file-like object)
                            # do_constant_folding=True,  # whether to execute constant folding for optimization
                            # input_names=['input'],  # the model's input names
                            input_names=['hidden_states', 'attention_mask', 'position_ids', 'key_cache', 'value_cache'],
                            output_names=['output', 'key_cache_out', 'value_cache_out'],  # the model's output names
                            # dynamic_axes={'input': {0 : 'batch_size', 1: 'num_tokens'},    # variable length axes
                            #                 'output' : {0 : 'batch_size', 1: 'num_tokens'}}
                          )
        break


print('Finished exporting model.')
