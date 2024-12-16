import streamlit as st
import requests

# FastAPI backend URL
FASTAPI_URL = "http://192.168.75.114:8000/chat"

# Streamlit app setup
st.set_page_config(page_title="Chat with Llama-2", layout="centered")
st.title("Chat with Llama-2")

# Session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "session1"  # Default session ID

# Chat input form
with st.form(key="chat_form"):
    user_message = st.text_input("Your Message:", "")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_message:
    # Send the message to FastAPI
    payload = {"user_message": user_message, "session_id": st.session_state.session_id}
    response = requests.post(FASTAPI_URL, json=payload)

    if response.status_code == 200:
        assistant_response = response.json().get("assistant_response", "No response")
        st.session_state.chat_history.append(("User", user_message))
        st.session_state.chat_history.append(("Assistant", assistant_response))
    else:
        st.error("Failed to connect to the API.")

# Display the chat history
for sender, message in st.session_state.chat_history:
    if sender == "User":
        st.markdown(f"**You:** {message}")
    elif sender == "Assistant":
        st.markdown(f"**Assistant:** {message}")
