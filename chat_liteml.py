from pydantic import BaseModel
# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer
from ceva_modeling_llama import LlamaForCausalLM, LlamaDecoderLayer
import torch
from liteml.ailabs_liteml.retrainer import RetrainerConfig, RetrainerModel
from liteml.ailabs_shared.load_config import load_config
from utils import get_calibration_loader
from torch.nn.functional import pad
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the input schema
class ChatRequest(BaseModel):
    user_message: str
    session_id: str  # Unique identifier for the chat session
    max_length: int = 4096  # Optional parameter with default


# Load the Llama-2-7b-hf model and tokenizer
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
# config_name = 'float'
config_name = 'configs/Eli/static_roni_w8a8ptok_sym_matmul_token_no_head_tensor.yaml' # The mixed dynamic and static conf

try:
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME, padding_side='right', padding=True)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"  # Automatically map the model to available GPUs/CPU
    )
    if '<pad>' not in tokenizer.get_vocab():
        # Add the pad token
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # Resize the embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Set embeddings of pad token to zero
    model.model.embed_tokens.weight.data[-1] = torch.zeros_like(model.model.embed_tokens.weight[-1])

    # Configure the pad token in the model
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    # Check if they are equal
    assert model.config.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"

    # Quantize model
    if config_name != 'float':
        conf = load_config(config_name)
        if 'SmoothQuant' in conf:
            conf["SmoothQuant"]["decoder_class"] = LlamaDecoderLayer
        if conf['QAT']['data_quantization']['quantization_mode'] == 'static':
            # Add calibration loader
            calib_loader = get_calibration_loader(tokenizer, seq_len=1024)
            conf["QAT"]["data_quantization"][
                "calibration_loader"
            ] = calib_loader
            conf["QAT"]["data_quantization"][
                "calibration_loader_key"
            ] = lambda model, x: model(x.cuda())
        with torch.no_grad():
            model = RetrainerModel(model, config=RetrainerConfig(conf))
        if 'OmniQuant' in conf["QAT"]:
            model = model.to(device)

except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load the model and tokenizer.")

# Store chat histories in memory (for simplicity)
chat_sessions = {}

def chat(request: ChatRequest):
    """
    Generate a chat response based on the user's message and session history.
    """
    # Retrieve or initialize chat history
    session_id = request.session_id
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    # Update chat history with the user message
    chat_history = chat_sessions[session_id]
    chat_history.append(f"User: {request.user_message}")

    # Create the conversation input
    conversation = "\n".join(chat_history) + "\nAssistant:"
    inputs = tokenizer(conversation, return_tensors="pt", padding=True).to(model.device)

    input_seq_len = inputs['input_ids'].shape[-1]
    # Pad input_ids and attention_mask
    inputs['input_ids'] = pad(inputs['input_ids'], (0, 1024 - input_seq_len), value=tokenizer.pad_token_id)
    inputs['attention_mask'] = pad(inputs['attention_mask'], (0, 1024 - input_seq_len), value=0)

    # Generate the model's response
    outputs = model.generate(
        **inputs,
        max_length=request.max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the assistant's response from the generated text
    assistant_response = generated_text[len(conversation):].strip()
    chat_history.append(f"Assistant: {assistant_response}")

    # Update the session history
    chat_sessions[session_id] = chat_history

    return {"assistant_response": assistant_response}


def main():
    print("Welcome to Llama chat with LiteML")
    while True:
        # get prompt from user
        print("User:")
        prompt = input()
        # create request
        req = ChatRequest(user_message=prompt, session_id="chat1")
        # send request to chat
        response = chat(req)
        print("Assistant:")
        print(response['assistant_response'])


if __name__ == "__main__":
    main()