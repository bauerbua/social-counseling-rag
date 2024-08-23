import os
import pdfplumber
from haystack.document_stores import ChromaDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.schema import Document
from sentence_transformers import SentenceTransformer
import chromadb

# Load environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize ChromaDB client
chroma_client = chromadb.Client()  # Initialize ChromaDB client
document_store = ChromaDocumentStore(client=chroma_client, embedding_dim=384)

# Initialize the embedding model for the retriever
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
retriever = EmbeddingRetriever(document_store=document_store, embedding_model=embedding_model)

# Function to extract text from a PDF and store it in ChromaDB
def load_and_index_pdf(pdf_path: str):
    with pdfplumber.open(pdf_path) as pdf:
        texts = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:  # Only add non-empty text
                texts.append(text)
    
    # Convert extracted text to Document objects
    documents = [Document(content=text) for text in texts]
    
    # Write documents to the document store
    document_store.write_documents(documents)
    
    # Update embeddings for the stored documents
    document_store.update_embeddings(retriever)

# Function to retrieve documents based on a query
def retrieve_documents(query: str):
    retrieved_docs = retriever.retrieve(query=query)
    return " ".join([doc.content for doc in retrieved_docs])

# Function to query ChromaDB and retrieve relevant information
def query_chromadb(query: str):
    relevant_passages = retrieve_documents(query)
    return relevant_passages

# Example usage
if __name__ == "__main__":
    # Path to the PDF file
    pdf_path = "sample.pdf"  # Update this path to 
