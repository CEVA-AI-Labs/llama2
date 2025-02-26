import argparse
import torch
from transformers import LlamaTokenizer
from ceva_modeling_llama_v4_46 import LlamaForCausalLM, LlamaDecoderLayer
from liteml.ailabs_liteml.retrainer import RetrainerConfig, RetrainerModel
from liteml.ailabs_shared.load_config import load_config
from utils import evaluate, get_calibration_loader


def load_liteml_spinquant_scales(model, liteml_spinquant_path):
    spinquant_state_dict = torch.load(liteml_spinquant_path)
    model.load_state_dict(spinquant_state_dict, strict=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the perplexity of the quantized Llama-2-7b model with different quantization configurations.")
    parser.add_argument('--model_dir', type=str, default='meta-llama/Llama-2-7b-hf', help='Directory of the pretrained model')
    parser.add_argument('--seq_len', type=int, default=2048, help='Sequence length for evaluation')
    parser.add_argument('--config_file', type=str, default='float', help='path to LiteML configuration file')
    parser.add_argument('--save_model_path', type=str, help='Path to save the retrained LiteML model')
    parser.add_argument('--load_model_path', type=str, help='Path to load the retrained LiteML model')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.load_model_path and args.load_spinquant_path:
        raise ValueError("Both load_model_path and load_spinquant_path cannot be provided at the same time.")
    print('Initializing tokenizer')
    tokenizer = LlamaTokenizer.from_pretrained(args.model_dir)
    print(f'Loading model: {args.config_file}')
    model = LlamaForCausalLM.from_pretrained(args.model_dir, device_map='auto', torch_dtype=torch.float16)

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
            if args.save_model_path:
                torch.save(model.state_dict(), args.save_model_path)

        ppl = evaluate(model, tokenizer, seq_len=args.seq_len)
        print(f'Model {args.config_file}')
        print(f'Perplexity: {ppl:.4}')
