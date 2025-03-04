from pprint import pprint
# from transformers import LlamaForCausalLM
from ceva_modeling_llama_v4_46 import LlamaForCausalLM, LlamaDecoderLayer
from transformers import LlamaTokenizer
from liteml.ailabs_liteml.retrainer import RetrainerConfig, RetrainerModel
from liteml.ailabs_shared.load_config import load_config
from utils import evaluate, get_calibration_loader
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
import torch

task_name = "wikitext"  # hellaswag
model_id = "meta-llama/Llama-2-7b-hf"
# config_file = "configs/w8a8_per_tensor_per_token_dynamic.yaml"
config_file = "float"
seq_len = 2048
model = LlamaForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16, attn_implementation="eager")
tokenizer = LlamaTokenizer.from_pretrained(model_id)

with torch.no_grad():
    if config_file != 'float':
        conf = load_config(config_file)
        if 'SmoothQuant' in conf:
            conf["SmoothQuant"]["decoder_class"] = LlamaDecoderLayer
        if conf['QAT']['data_quantization']['quantization_mode'] == 'static':
            calib_loader = get_calibration_loader(tokenizer, seq_len=seq_len)
            conf["QAT"]["data_quantization"]["calibration_loader"] = calib_loader
            conf["QAT"]["data_quantization"]["calibration_loader_key"] = lambda model, x: model(x.cuda())

        model = RetrainerModel(model, config=RetrainerConfig(conf))

hf_model = HFLM(model)

results = simple_evaluate(hf_model, tasks=[task_name], batch_size=8)

pprint(results['results'])