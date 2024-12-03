from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define the input schema
class ChatRequest(BaseModel):
    user_message: str
    session_id: str  # Unique identifier for the chat session
    max_length: int = 2048  # Optional parameter with default

# Initialize the FastAPI app
app = FastAPI()

# Load the Llama-2-7b-hf model and tokenizer
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"  # Automatically map the model to available GPUs/CPU
    )
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load the model and tokenizer.")

# Store chat histories in memory (for simplicity)
chat_sessions = {}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Llama-2-7b-hf Chat API"}

@app.post("/chat")
def chat(request: ChatRequest):
    """
    Generate a chat response based on the user's message and session history.
    """
    try:
        # Retrieve or initialize chat history
        session_id = request.session_id
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []

        # Update chat history with the user message
        chat_history = chat_sessions[session_id]
        chat_history.append(f"User: {request.user_message}")

        # Create the conversation input
        conversation = "\n".join(chat_history) + "\nAssistant:"
        inputs = tokenizer(conversation, return_tensors="pt").to(model.device)

        # Generate the model's response
        outputs = model.generate(
            **inputs,
            max_length=request.max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the assistant's response from the generated text
        assistant_response = generated_text[len(conversation):].strip()
        chat_history.append(f"Assistant: {assistant_response}")

        # Update the session history
        chat_sessions[session_id] = chat_history

        return {"assistant_response": assistant_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during chat: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="192.168.75.114", port=8000)  # for gpu 1
    uvicorn.run(app, host="192.168.75.63", port=8000)  # for gpu 3
