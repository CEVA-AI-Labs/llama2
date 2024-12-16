# Chat Application with Llama-2

This project consists of a FastAPI backend and a Streamlit frontend for a chat application using the Llama-2-7b model.

## Prerequisites

- Python 3.8 or higher
- install LiteML requirements, upgrade transformers to v4.46.2, install fastapi and streamlit
  ```
  cd ../../ailabs_liteml
  pip install  --upgrade -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
  pip install  --upgrade -r requirements_test.txt
  pip install transformers==4.46.2
  pip install "fastapi[standard]"
  pip install streamlit
   ```


## 1. Running the FastAPI Backend
Open terminal, change directory to chat_api and run the following code where SERVER_IP is the IP of host server.

   ```
   # Run the FastAPI server
   python api_main_chat_liteml.py --ip SERVER_IP
   ```

## 2. Running the Streamlit Frontend
Open a new terminal and run the code below.
   ```
   # Run the Streamlit application
   # Set the IP address as an environment variable
   export FASTAPI_IP=SERVER_IP

   # Run the Streamlit app
   streamlit run streamlit_app.py --ip $FASTAPI_IP
   ```
