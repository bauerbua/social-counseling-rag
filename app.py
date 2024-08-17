import os
import chainlit as cl
from transformers import pipeline

# Load the Hugging Face API token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

# Runs when the chat starts
@cl.on_chat_start
def main():
    # Create the pipeline using the Mistral model on Hugging Face
    llm = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.1",
        tokenizer="mistralai/Mistral-7B-Instruct-v0.1",
        use_auth_token=HF_TOKEN,
        device=0  # Set to -1 for CPU, 0 or higher for GPU
    )

    # Store the llm in the user session
    cl.user_session.set("llm", llm)
    
# Runs when a message is sent
@cl.on_message
async def main(message: cl.Message):
    # Retrieve the model from the user session
    llm = cl.user_session.get("llm")

    # Prepare the prompt for the model
    prompt = f"[INST]{message.content}[/INST]"

    # Generate the response using the pipeline
    response = llm(prompt, max_new_tokens=100, temperature=0.7)

    # Stream the response back to the user
    msg = cl.Message(content="")
    for text in response[0]['generated_text']:
        await msg.stream_token(text)

    await msg.send()
