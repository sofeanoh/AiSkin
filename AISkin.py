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

#%% Constants
BASE_API_URL = "https://91d4-2001-e68-5431-ce6f-b5af-bfc2-7b36-e93e.ngrok-free.app"
FLOW_ID = "b62a6fd3-be02-4490-84b2-2374a84e66c2"
ENDPOINT = "AiSkin" # The endpoint name of the flow

#%% Model loading
# Load the PyTorch model
model = YOLO('best.pt')
model.eval()

def run_flow(message: str, endpoint: str, output_type: str = "chat", input_type: str = "chat", history: list = []) -> dict:
    """
    Run a flow with a given message and optional tweaks.

    :param message: The message to send to the flow
    :param endpoint: The ID or the endpoint name of the flow
    :param output_type: The type of output expected
    :param input_type: The type of input provided
    :param history: The conversation history to include
    :return: The JSON response from the flow
    """
    api_url = f"{BASE_API_URL}/api/v1/run/{endpoint}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
        "history": history
    }
    headers = None
    response = requests.post(api_url, json=payload, headers=headers)
    # Log the response for debugging purpose
    logging.info(f"Response Status Code: {response.status_code}")
    logging.info(f"Response Text: {response.text}")
    return response.json()

def extract_message(response: dict) -> str:
    try:
        # Extract response message
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        logging.error("No valid message found in the response")
        return "No valid message found in the response."

# Predict the image
def predict_image(image_data):
    # Convert the image to JPEG format
    image = Image.open(io.BytesIO(image_data.getbuffer()))
    jpeg_image = io.BytesIO()
    image.save(jpeg_image, format='JPEG')
    jpeg_image.seek(0) # Convert to PIL Image
    pil_image = Image.open(jpeg_image) # Run prediction
    results = model(pil_image) # Extract the labels
    labels = []
    for result in results:
        for box in result.boxes:
            labels.append(result.names[box.cls.item()])
    # Debug: Display labels
    for label in labels:
        st.write(label)
    # Concatenate labels into a single string
    labels_string = ', '.join(labels)
    # Display labels string
    return labels_string # Return the concatenated labels string


def main():
    st.title("🌸 AICare4UrSkin 🌸")
    st.markdown("#### Discover insights about your skincare products with ease.")
    st.markdown("🚨🚨🚨 To ask further about your captured image, please clear the image first.")
    
    with st.sidebar:
        st.title('AI Skin Care Chatbot')
        st.markdown('This is a simple chatbot that helps you to find out more about your skin care product. You can either upload an image of your skincare product(s) or take a picture with your camera.')
        
        enable_camera = st.checkbox("Enable camera input")
        camera_picture = st.camera_input("Take a picture", disabled=not enable_camera)
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    # Initialize session state for chat history and image storage
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "input_image" not in st.session_state:
        st.session_state.input_image = None  # Store image separately
    
    # Display previous messages, whether images or text
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            if message.get("content_type", "text") == "image":  # Default to "text" if "content_type" is missing
                st.image(message["content"])  # Display image
            else:
                st.write(message["content"])  # Display text


    # Input box for user message (only text input visible after the image is processed)
    if not st.session_state.input_image:
        input_image = camera_picture if enable_camera and camera_picture else uploaded_file  # either camera or upload

        if input_image: 
            # Store the image in session state (to remember it across interactions)
            st.session_state.input_image = input_image

            # Add user message with image content_type
            st.session_state.messages.append(
                {"role": "user", 
                 "content": input_image,
                 "content_type": "image",  # content_type for image
                 "avatar": "🗯️"  # emoji for user
                } 
            )

            prediction = predict_image(input_image)

            with st.chat_message("user", avatar="🗯️"):  # Display user query
                st.image(input_image)
                
            # Call the Langflow API and get the assistant's response
            with st.chat_message("assistant", avatar="🤖"):  # emoji for assistant
                message_placeholder = st.empty()  # Placeholder for assistant response
                with st.spinner("Waiting for response..."):
                    assistant_response = extract_message(run_flow(prediction, endpoint=ENDPOINT))
                    message_placeholder.write(assistant_response)  # Add assistant response to session state
            
            # Add assistant response to session state
            st.session_state.messages.append(
                {"role": "assistant", 
                 "content": assistant_response,
                 "content_type": "text",  # content_type for text
                 "avatar": "🤖"  # emoji for assistant
                }
            )

    # Only show text input after the image is processed
    elif query := st.chat_input("Please provide your question or request here"):
        # Add user message to chat history with 'text' content type
        st.session_state.messages.append(
            {"role": "user", 
             "content": query,
             "content_type": "text",  # content_type for text
             "avatar": "🗯️"  # emoji for user
            }
        )
        
        with st.chat_message("user", avatar="🗯️"):  # Display user query
            st.write(query)
        
        # Call the Langflow API and get the assistant's response
        with st.chat_message("assistant", avatar="🤖"):  # emoji for avatar
            message_placeholder = st.empty()  # Placeholder for assistant response
            with st.spinner("Waiting for response..."):
                assistant_response = extract_message(run_flow(query, endpoint=ENDPOINT))
                message_placeholder.write(assistant_response)  # Display assistant response
        
        # Add assistant response to session state with 'text' content type
        st.session_state.messages.append(
            {"role": "assistant", 
             "content": assistant_response,
             "content_type": "text",  # content_type for text
             "avatar": "🤖"  # emoji for assistant
            }
        )


if __name__ == "__main__":
    main()
