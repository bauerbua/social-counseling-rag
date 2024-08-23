import os
import chainlit as cl
import requests
from haystack.document_stores import WeaviateDocumentStore
from haystack.nodes import EmbeddingRetriever

# Initialize WeaviateDocumentStore
document_store = WeaviateDocumentStore(
    host='http://localhost',
    port=8080,
    embedding_dim=384,
    index="Document",  # The class name in Weaviate where documents are stored
)

# Initialize EmbeddingRetriever
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the Hugging Face API token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

# Define the Mistral API endpoint and model
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

# Helper function to make API requests to the Mistral model
def query_mistral_model(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "options": {"use_cache": False}}
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # Raise an error for HTTP issues
    result = response.json()
    return result[0]['generated_text']

# Retrieve relevant documents based on a query
def retrieve_documents(query: str):
    retrieved_docs = retriever.retrieve(query=query)
    # Concatenate documents into one string for context
    return " ".join([doc.content for doc in retrieved_docs])

# Runs when the chat starts
@cl.on_chat_start
def on_chat_start():
    pass  # No need to load and index the PDF here

# Runs when a message is sent
@cl.on_message
async def on_message(message: cl.Message):
    query = message.content

    try:
        # Step 1: Retrieve relevant passages using embeddings
        relevant_passages = retrieve_documents(query)

        # Step 2: Formulate a prompt for the Mistral model
        prompt = f"Use the following information to answer the question, use only the provided document. Do not include any of the context or the prompt in your response.\n\nContext:\n{relevant_passages}\n\nQuestion: {query}\nAnswer:"

        # Step 3: Generate the answer using the Mistral model
        result = query_mistral_model(prompt)

        # Process the result to extract only the answer
        answer = result.split("Answer:")[-1].strip()

        # Send only the result back to the user
        await cl.Message(author="Bot", content=answer).send()
    except Exception as e:
        await cl.Message(author="Bot", content=f"Error: {str(e)}").send()
