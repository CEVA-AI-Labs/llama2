from pprint import pprint
# from transformers import LlamaForCausalLM
from ceva_modeling_llama_v4_46 import LlamaForCausalLM, LlamaDecoderLayer
from transformers import LlamaTokenizer
from liteml.ailabs_liteml.retrainer import RetrainerConfig, RetrainerModel
from liteml.ailabs_shared.load_config import load_config
from utils import get_calibration_loader
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
import torch
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the perplexity of the quantized Llama-2-7b model with "
                                                 "different quantization configurations.")
    parser.add_argument('--model_dir', type=str, default='meta-llama/Llama-2-7b-hf', help='Directory of the '
                                                                                          'pretrained model')
    parser.add_argument('--config_file', type=str, default='float', help='path to LiteML configuration file')
    parser.add_argument('--tasks', nargs='+', default=["arc_easy", "hellaswag", "piqa", "openbookqa", "winogrande"],
                        help='List of evaluation task names (e.g., arc_easy hellaswag piqa)')
    parser.add_argument('--seq_len', type=int, default=2048, help='Sequence length for calibration')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--num_fewshot', type=int, default=None, help='Number of examples in few-shot context')
    parser.add_argument('--limit', type=float, default=None, help='Limit the number of examples per task (only use this '
                                                               'for testing), If <1, limit is a percentage of the '
                                                               'total number of examples.')
    parser.add_argument('--load_model_path', type=str, help='Path to load the retrained LiteML model')
    parser.add_argument('--results_path', type=str, default='lm-eval_results.json', help='Output path of the results file.')
    return parser.parse_args()


def main():
    args = parse_args()
    model = LlamaForCausalLM.from_pretrained(args.model_dir, device_map='auto', torch_dtype=torch.float16, attn_implementation="eager")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_dir)

    with torch.no_grad():
        if args.config_file != 'float':
            conf = load_config(args.config_file)
            if 'SmoothQuant' in conf:
                conf["SmoothQuant"]["decoder_class"] = LlamaDecoderLayer
            if conf['QAT']['data_quantization']['quantization_mode'] == 'static':
                calib_loader = get_calibration_loader(tokenizer, seq_len=args.seq_len)
                conf["QAT"]["data_quantization"]["calibration_loader"] = calib_loader
                conf["QAT"]["data_quantization"]["calibration_loader_key"] = lambda model, x: model(x.cuda())
            if args.load_model_path:
                model = RetrainerModel.from_pretrained(model,
                                                       args.config_file,
                                                       args.load_model_path,
                                                       device=None,
                                                       dummy_input=torch.randint(0, 32000, (1, args.seq_len)),
                                                       strict=False,
                                                       map_location=lambda storage, loc: storage)
            else:
                model = RetrainerModel(model, config=RetrainerConfig(conf))

    hf_model = HFLM(model)
    results = simple_evaluate(hf_model, tasks=args.tasks, batch_size=args.batch_size, num_fewshot=args.num_fewshot, limit=args.limit)
    pprint(results['results'])

    with open(args.results_path, 'w') as fp:
        json.dump(results['results'], fp)  # numbers only


if __name__ == '__main__':
    main()
