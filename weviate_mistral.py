import os
import pdfplumber
import chainlit as cl
import requests
from haystack.document_stores  import WeaviateDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.schema import Document
import weaviate

 
# Initialize WeaviateDocumentStore
document_store = WeaviateDocumentStore(
    host='http://localhost',
    port=8080,
    #client=weaviate_client, 
    embedding_dim=384,
    index="Document",  # The class name in Weaviate where documents will be stored
    duplicate_documents="overwrite"  # Handling of duplicate documents
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

# Load the PDF file and create embeddings
def load_and_index_pdf(pdf_path: str):
    with pdfplumber.open(pdf_path) as pdf:
        texts = []
        for page in pdf.pages:
            texts.append(page.extract_text())
    documents = [Document(content=text) for text in texts if text]
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)


# Retrieve relevant documents based on a query
def retrieve_documents(query: str):
    retrieved_docs = retriever.retrieve(query=query)
    # Concatenate documents into one string for context
    return " ".join([doc.content for doc in retrieved_docs])

# Runs when the chat starts
@cl.on_chat_start
def on_chat_start():
    # Load and index a sample PDF
    pdf_path = "paper.pdf"  # Update this path
    load_and_index_pdf(pdf_path)

# Runs when a message is sent
@cl.on_message
async def on_message(message: cl.Message):
    query = message.content

    try:
        # Step 1: Retrieve relevant passages using embeddings
        relevant_passages = retrieve_documents(query)

     
        # Step 2: Formulate a prompt for the Mistral model
        # The prompt now instructs the model to provide only the answer
        prompt = f"Use the following information to answer the question, use only the provided document. Do not include any of the context or the prompt in your response.\n\nContext:\n{relevant_passages}\n\nQuestion: {query}\nAnswer:"

        # Step 3: Generate the answer using the Mistral model
        result = query_mistral_model(prompt)

        # Process the result to extract only the answer (if needed)
        # Assuming the result might include extra text, extract only the answer part
        answer = result.split("Answer:")[-1].strip()

        # Send only the result back to the user
        await cl.Message(author="Bot", content=answer).send()
    except Exception as e:
        await cl.Message(author="Bot", content=f"Error: {str(e)}").send()
