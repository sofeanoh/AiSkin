#%% import libraries
import streamlit as st
import requests
import logging
import cv2, numpy as np
import base64
from PIL import Image
import io
import torch
from ultralytics import YOLO


BASE_API_URL = "https://3c1a-2001-e68-5431-4c14-5d9d-fa47-53f9-94a7.ngrok-free.app"
FLOW_ID = "b62a6fd3-be02-4490-84b2-2374a84e66c2"
ENDPOINT = "AiSkin" # The endpoint name of the flow

#%% Model loading
# Load the YOLO model
model = YOLO('best.pt')
model.eval()

#%%
def run_flow(message: str,
  endpoint: str,
  output_type: str = "chat",
  input_type: str = "chat",
  history: list = []):
           
  """
  Run a flow with a given message and optional tweaks.

  :param message: The message to send to the flow
  :param endpoint: The ID or the endpoint name of the flow
  :param tweaks: Optional tweaks to customize the flow
  :return: The JSON response from the flow
  """
  api_url = f"{BASE_API_URL}/api/v1/run/{endpoint}"

  payload = {
      "input_value": message,
      "output_type": output_type,
      "input_type": input_type,
      "history": history,
  }
  headers = None
  response = requests.post(api_url, json=payload, headers=headers)
  # Log the response for debugging purpose
  logging.info(f"Response Status Code: {response.status_code}")
  logging.info(f"Response Text: {response.text}")
  return response.json()


def extract_message(response: dict) -> str:
    try:
        # for extracting response message
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        logging.error("No valid message found in the response")
        return "No valid message found in the response."

#%% Predict the image
def predict_image(image_data):
    # need to convert the image to JPEG format first
    image = Image.open(io.BytesIO(image_data.getbuffer()))
    jpeg_image = io.BytesIO()
    image.save(jpeg_image, format='JPEG')
    jpeg_image.seek(0) # convert to PIL Image
    pil_image = Image.open(jpeg_image) 
    
    # Run prediction
    results = model(pil_image) 
    
    # etract the labels
    labels = []
    for result in results:
        for box in result.boxes:
            labels.append(result.names[box.cls.item()])
    #Display labels
    for label in labels:
        st.write(label)
    #concatenate labels into a single string bcs we want the langflow model to take input as a string
    labels_string = ', '.join(labels)
    return labels_string 


def main():
    st.title("🌸 AICare4UrSkin 🌸")
    st.markdown("#### Discover insights about your skincare products with ease.")
    st.markdown("🚨🚨🚨 To ask further about your captured image, please clear the image first.")
    
    with st.sidebar:
        st.title('AI Skin Care Chatbot')
        st.markdown('This is a simple chatbot that helps you to find out more about your skincare product. You can either upload an image of your skincare product(s) or take a picture with your camera.')
        
        enable_camera = st.checkbox("Enable camera input")
        camera_picture = st.camera_input("Take a picture", disabled=not enable_camera)
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Initialize session state for chat history and image storage
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "input_image" not in st.session_state:
        st.session_state.input_image = None  # Store image separately
    if "product_info" not in st.session_state:
        st.session_state.product_info = None  # Store product info (e.g., labels or product name)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            if message.get("content_type", "text") == "image":
                st.image(message["content"])
            else:
                st.write(message["content"])
    
    # Handle text input (chat query first)
    query = st.chat_input("Ask your question or type your request here:")
    if query:
        # Clear image input if there's a chat query
        st.session_state.input_image = None
        st.session_state.image_processed = False
        st.session_state.camera_picture = None
        st.session_state.uploaded_file = None

        # Add user query to chat history
        st.session_state.messages.append(
            {"role": "user", "content": query, "content_type": "text", "avatar": "🗯️"}
        )

        with st.chat_message("user", avatar="🗯️"):
            st.write(query)

        # Include product info in history if available
        history = [{"role": "user", "content": f"Product info: {st.session_state.product_info}"}] if st.session_state.product_info else []
        
        # Get assistant response for the query
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                assistant_response = extract_message(run_flow(query, endpoint=ENDPOINT, history=history))
                st.write(assistant_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response, "content_type": "text", "avatar": "🤖"}
        )

    # Handle image input (only if no chat input was made)
    input_image = camera_picture if enable_camera and camera_picture else uploaded_file
    if input_image and not st.session_state.input_image:
        # Process the image
        st.session_state.input_image = input_image
        prediction = predict_image(input_image)

        # Store product info (labels or product name)
        st.session_state.product_info = prediction

        # Add the image to the chat history
        st.session_state.messages.append(
            {"role": "user", "content": input_image, "content_type": "image", "avatar": "🗯️"}
        )

        with st.chat_message("user", avatar="🗯️"):
            st.image(input_image)

        # Get the assistant's response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyzing image..."):
                assistant_response = extract_message(run_flow(prediction, endpoint=ENDPOINT))
                st.write(assistant_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response, "content_type": "text", "avatar": "🤖"}
        )

if __name__ == "__main__":
    main()
