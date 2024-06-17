from transformers import OPTForCausalLM, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from huggingface_hub import login
import torch
from liteml.ailabs_shared.omniquant.omniquant import OmniQuantModel
import os
from liteml.ailabs_shared.load_config import load_config
from liteml.ailabs_liteml.retrainer import RetrainerModel, RetrainerConfig
from datasets import load_dataset
from tqdm import tqdm
from evaluate_models import evaluate
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    model_dir = 'meta-llama/Llama-2-7b-hf'
    print('Loading model')
    # model = OPTForCausalLM.from_pretrained(model_dir, device_map=0, torch_dtype=torch.float16)  # put the model on gpu 0

    # Like in OmniQuant
    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, config=config, device_map='cpu', torch_dtype=torch.float16)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # dataset = load_dataset('lambada', split='validation[:1000]')

    root = os.path.abspath(__file__).rsplit(os.path.sep, 1)[0]

    conf = load_config(os.path.join(root, "test_config_llama.yaml"))

    model_quantized = RetrainerModel(model, config=RetrainerConfig(conf)).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, legacy=False)
    ppl = evaluate(model_quantized, tokenizer, seq_len=2048)
    print(ppl)
    print('Done')


if __name__ == '__main__':
    main()
