import torch
import onnx
import onnxruntime as ort
from export_to_onnx_float import create_inputs
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer, OPTForCausalLM, AutoTokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_causal_mask(input_ids_shape):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = np.full((tgt_len, tgt_len), np.finfo(np.float32).min)
    mask_cond = np.arange(mask.shape[-1])
    mask = np.where(mask_cond < (mask_cond + 1).reshape(mask.shape[-1], 1), 0, mask)

    return np.expand_dims(np.expand_dims(mask, axis=0), axis=0).repeat(bsz, axis=0)


def expand_mask(mask, tgt_len=None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = np.expand_dims(mask, axis=[0, 1]).repeat(tgt_len, axis=2).astype(np.float32)

    inverted_mask = 1.0 - expanded_mask

    return np.where(inverted_mask.astype(bool), np.finfo(np.float32).min, inverted_mask)


def create_inputs_numpy(tokens):
    input_embeds = np.random.rand(1, tokens, 4096).astype(np.float32)
    attention_mask = np.ones((1, tokens))
    causal_mask = make_causal_mask((1, tokens))
    attention_mask = expand_mask(attention_mask, tgt_len=tokens)
    combined_mask = np.clip(attention_mask + causal_mask, np.finfo(np.float32).min, np.finfo(np.float32).max)

    return input_embeds, combined_mask


def compare_outputs(x1, x2):
    diff = x1 - x2
    abs_max_diff = np.max(np.abs(diff))
    mse = np.mean(np.square(diff))
    return abs_max_diff, mse

def main():
    model_dir = 'meta-llama/Llama-2-7b-hf'

    # hidden_states, attention_mask = create_inputs_numpy(tokens=1024)
    inputs = create_inputs(tokens=1024, stage='prompt', dtype=torch.float32, device=device)
    hidden_states = inputs['hidden_states'].detach().cpu().numpy().astype(np.float32)
    attention_mask = inputs['attention_mask'].detach().cpu().numpy().astype(np.float32)
    model_torch = LlamaForCausalLM.from_pretrained(model_dir, device_map='auto', torch_dtype=torch.float32)
    first_decoder_block = model_torch.model.layers[0]
    torch_output, torch_kv_cache = first_decoder_block(hidden_states=inputs['hidden_states'],
                                        attention_mask=inputs['attention_mask'],
                                        position_ids=inputs['position_ids'],
                                        past_key_value=inputs['past_key_value'],
                                        use_cache=inputs['use_cache']
                                        )
    torch_output = torch_output.detach().cpu().numpy()
    torch_key_cache_out = torch_kv_cache[0].detach().cpu().numpy()
    torch_value_cache_out = torch_kv_cache[1].detach().cpu().numpy()
    ort_sess = ort.InferenceSession('onnx/llama2_prompt_phase_optimized.onnx')

    onnx_output, onnx_key_cache_out, onnx_value_cache_out = ort_sess.run(['output', 'key_cache_out', 'value_cache_out'], {"hidden_states": hidden_states, "attention_mask": attention_mask})
    abs_max_diff_output, mse_output = compare_outputs(torch_output, onnx_output)
    abs_max_diff_key_cache, mse_key_cache = compare_outputs(torch_key_cache_out, onnx_key_cache_out)
    abs_max_diff_value_cache, mse_value_cache = compare_outputs(torch_value_cache_out, onnx_value_cache_out)
    print(f'[Output] abs max diff: {abs_max_diff_output}, MSE: {mse_output}')
    print(f'[Key cache] abs max diff: {abs_max_diff_key_cache}, MSE: {mse_key_cache}')
    print(f'[Value cache] abs max diff: {abs_max_diff_value_cache}, MSE: {mse_value_cache}')


if __name__ == '__main__':
    main()
