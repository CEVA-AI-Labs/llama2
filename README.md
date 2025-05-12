# Llama-2-7b Quantization Perplexity Evaluation

This repository contains a script to evaluate the perplexity of the quantized Llama-2-7b model with different quantization configurations using LiteML.

## Description

The script loads the Llama-2-7b model from HuggingFace and performs quantization using LiteML. It accepts a configuration file for LiteML representing a quantization scheme and calculates the perplexity of the model.

## Requirements

- Python 3.8+
- PyTorch
- Transformers 4.46
- LiteML


## Usage

Run the script with the desired arguments.

### Arguments

* **model_dir**: Directory of the pretrained model (default: 'meta-llama/Llama-2-7b-hf')
 
* **seq_len**: Sequence length for evaluation (default: 2048)

* **config_file**: Path to LiteML configuration file (default: 'float')

* **save_model_path**: Path to save the retrained LiteML model

* **load_model_path**: Path to load the retrained LiteML model

Below are some examples:


#### PTQ run with LiteML:
```bash
python evaluate_models.py --config_file configs/w8a8_npm_v1_3_4.yaml
```

#### Load weights and scale factors from SpinQuant's state dict and perform PTQ with LiteML:
```bash
python evaluate_models.py --config_file configs/spinquant/w4a8_liteml_spinquant_e_external_state_dict.yaml
```

#### Load weights and scale factors from SpinQuant's state dict, perform PTQ using LiteML and save the model's state dict after calibration:
```bash
python evaluate_models.py --config_file configs/spinquant/w4a8_liteml_spinquant_e_external_state_dict.yaml --save_model_path /path/to/liteml_spinquant.pth
```

#### Load a pretrained LiteML model and perform PTQ (skips the calibration process):
```bash
python evaluate_models.py --config_file configs/spinquant/w4a8_liteml_spinquant_e_external_state_dict.yaml --load_model_path /path/to/liteml_spinquant.pth
```

#### Load weights and scale factors from SpinQuant's state dict and perform PTQ with LiteML using "true quant" mode:
```bash
python evaluate_models.py --config_file configs/spinquant/w4a8_liteml_spinquant_truequant_pwla.yaml
```

