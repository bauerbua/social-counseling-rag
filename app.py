import os
import chainlit as cl
from main import queryDB

@cl.on_message
async def handle_message(message: cl.Message):
    """Handles user messages and returns responses from the Haystack pipeline."""

    # Run the pipeline
    response = queryDB(message.content)

    # Extract LLM response
    reply = response["replies"][0].text

    # Send response back to Chainlit UI
    await cl.Message(content=reply).send()
