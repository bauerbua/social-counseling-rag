import os
import chainlit as cl
import requests

# Load the Hugging Face API token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

# Define the API endpoint and model
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

# Helper function to make API requests
def query_model(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "options": {"use_cache": False}}
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # Raise an error for HTTP issues
    result = response.json()
    return result[0]['generated_text']

# Runs when the chat starts
@cl.on_chat_start
def on_chat_start():
    # Initialization or setup if needed
    pass

# Runs when a message is senhttp://localhost:8000/
async def on_message(message: cl.Message):
    # Prepare the prompt
    prompt = f"[INST]{message.content}[/INST]"

    # Query the model
    try:http://localhost:8000/
        result = query_model(prompt)

        # Filter the output to remove the prompt or any undesired text
        filtered_result = result.replace(prompt, "").strip()  # Remove the prompt from the result

        # Optional: If there's still extra text you don't want, you can further refine this step.

        await cl.Message(author="Bot", content=filtered_result).send()
    except Exception as e:
        await cl.Message(author="Bot", content=f"Error: {str(e)}").send()
