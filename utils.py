import torch
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import torch.nn as nn


def get_calibration_loader(tokenizer: LlamaTokenizer, seq_len: int = 2048, split_percentage=10) -> list:
    """
    Loads calibration dataset from wikitext that is used to calibrate the quantization parameters during static
    quantization. The data is obtained from wikitext2 dataset.
    Args:
        tokenizer (LlamaTokenizer): A tokenizer suitable for Llama-2 model.
        seq_len (int): Sequence length of each batch in the calibration data.
        split_percentage (int): Percentage of the dataset to use for calibration, between 0 and 100.

    Returns:
        List of tensors of shape (1, seq_len) containing the input tokens.

    """
    calib = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[0:{split_percentage}%]")
    calibloader = tokenizer("\n\n".join(calib["text"]), return_tensors="pt")

    encodings = calibloader.input_ids
    nsamples = encodings.numel() // seq_len
    batch = [encodings[:, (i * seq_len): ((i + 1) * seq_len)] for i in range(nsamples)]
    return batch


def evaluate(model: LlamaForCausalLM, tokenizer: LlamaTokenizer, seq_len=2048, split_percentage=100) -> torch.Tensor:
    """
    Evaluate model perplexity on wikitext-2 test dataset.
    Args:
        model: Llama model. Can be wrapped with LiteML RetrainerModel.
        tokenizer (LlamaTokenizer): A tokenizer suitable for Llama-2 model.
        seq_len (int): Sequence length of each batch in the evaluation data.
        split_percentage (int): Percentage of the dataset to use for evaluation, between 0 and 100.

    Returns:
        Perplexity of the model on wikitext-2 test dataset.

    """
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"test[0:{split_percentage}%]")
    testloader = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    encodings = testloader.input_ids
    nsamples = encodings.numel() // seq_len
    nlls = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(nsamples)):
            batch = encodings[:, (i * seq_len): ((i + 1) * seq_len)].to(model.device)
            outputs = model(batch)
            logits = outputs[0]
            shift_logits = logits[:, :-1, :]
            shift_labels = encodings[:, (i * seq_len): ((i + 1) * seq_len)][:, 1:].to(model.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * seq_len
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seq_len))
    return ppl