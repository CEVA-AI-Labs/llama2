"""
A chatbot with Llama-2 model. We would like to replace the model with a quantized Llama model.
"""
from transformers import pipeline, Conversation

model_dir = 'meta-llama/Llama-2-7b-chat-hf'
chatbot = pipeline("conversational", model=model_dir, device_map='auto')
while True:
    inputs = input('>')
    speaker = Conversation(inputs)
    response = chatbot(speaker)
    print(response.generated_responses[-1])