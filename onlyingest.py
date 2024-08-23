import os
import pdfplumber
from haystack.document_stores import WeaviateDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.schema import Document
import weaviate

# Initialize Weaviate client
weaviate_client = weaviate.Client("http://localhost:8080")

# Initialize WeaviateDocumentStore
document_store = WeaviateDocumentStore(
    host='http://localhost',
    port=8080,
    embedding_dim=384,
    index="Document",  # The class name in Weaviate where documents will be stored
    duplicate_documents="overwrite"  # Handling of duplicate documents
)

# Initialize EmbeddingRetriever
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the PDF file and create embeddings
def load_and_index_pdf(pdf_path: str):
    with pdfplumber.open(pdf_path) as pdf:
        texts = []
        for page in pdf.pages:
            texts.append(page.extract_text())
    documents = [Document(content=text) for text in texts if text]
    document_store.write_documents(documents)
    document_store.update_embeddings(retriever)

if __name__ == "__main__":
    # Load and index a sample PDF
    pdf_path = "paper.pdf"  # Update this path
    load_and_index_pdf(pdf_path)
