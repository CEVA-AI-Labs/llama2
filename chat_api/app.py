import streamlit as st
import requests
import os


def main():
    # Get IP address from environment variable
    ip_address = os.getenv("FASTAPI_IP")
    if not ip_address:
        st.error("IP address not set. Please set the FASTAPI_IP environment variable.")
        return

    # FastAPI backend URL
    FASTAPI_URL = f"http://{ip_address}:8000/chat"
    RESET_ROUTE = f"http://{ip_address}:8000/reset"

    # Streamlit app setup
    st.set_page_config(page_title="Chat with Llama-2", layout="centered")
    st.title("Chat with Llama-2")

    # Session state to store chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = "session1"  # Default session ID
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0

    reset_button = st.sidebar.button("Reset", type="primary")

    # Chat input form
    prompt = st.chat_input("Say something")

    if prompt:
        # Send the message to FastAPI
        payload = {"user_message": prompt, "session_id": st.session_state.session_id}
        response = requests.post(FASTAPI_URL, json=payload)

        if response.status_code == 200:
            data = response.json()
            assistant_response = data.get("assistant_response", "No response")
            st.session_state.token_count = data.get("token_count")
            st.session_state.chat_history.append(("User", prompt))
            st.session_state.chat_history.append(("Assistant", assistant_response))
        else:
            st.error("Failed to connect to the API.")

    if reset_button:
        response = requests.post(RESET_ROUTE)
        if response.status_code == 200:
            data = response.json()
            messages = data.get("messages", None)
            st.session_state.token_count = data.get("token_count")
            st.session_state.chat_history = []
        else:
            st.error("Failed to connect to the API.")

    # Display the chat history
    for sender, message in st.session_state.chat_history:
        if sender == "User":
            st.markdown(f"**You:** {message}")
        elif sender == "Assistant":
            st.markdown(f"**Assistant:** {message}")

    # Token count display
    st.sidebar.write(f"**Total Tokens Used:** {st.session_state['token_count']}/4096")


if __name__ == "__main__":
    main()
