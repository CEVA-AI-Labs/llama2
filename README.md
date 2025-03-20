# Llama-2-7b Quantization Perplexity Evaluation

This repository contains a script to evaluate the perplexity of the quantized Llama-2-7b model with LiteML, using [SpinQuant](https://github.com/facebookresearch/SpinQuant) weights.

## Description

The script loads the Llama-2-7b model from HuggingFace and then loads the weights and scale factors obtained from running SpinQuant with W4A8 configuration.
The script then performs evaluation over wikitext-2 datasets.

## Installation
1. Copy the provided wheel file to this folder
2. Create virtual environment with python=3.8 and activate it
```bash
python3.8 -m venv venv
source venv/bin/activate
```
3. Install wheel and requirements
```bash
pip install liteml-25.0.0-cp38-cp38-linux_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu117
pip install --upgrade -r requirements.txt
```

## Usage

#### Selecting quantization configuration:
In the config_list in evaluate_models.py uncomment the desired configuration. The following configurations can be used:
* 'float' - to run the float16 model (wihtout quantization) set enable_spinquant = False in the evaluate_models.py script.
* 'configs/spinquant/w4a8_spinquant_PWLA.yaml'
* 'configs/spinquant/w4a16_spinquant_PWLA.yaml'

#### Set spinquant path
set spinquant_path variable in evaluate_models.py to the path of spinquant's pth file.

#### Running evaluation with SpinQuant:
```bash
python evaluate_models.py
```
