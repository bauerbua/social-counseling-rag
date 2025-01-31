
from pathlib import Path
from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.converters import PyPDFToDocument
from haystack import Pipeline
from haystack.document_stores.types import DuplicatePolicy

import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

"""
This script initializes a Weaviate client and a WeaviateDocumentStore to store and manage documents.
It defines a function `load_and_preprocess_files` that loads PDF files from a specified directory,
processes them through a series of preprocessing steps, and stores the resulting documents in the WeaviateDocumentStore.

The preprocessing steps include:
- Routing files based on their MIME type
- Converting PDF files to document objects
- Joining document parts
- Cleaning the documents
- Splitting documents into smaller chunks
- Embedding the documents using a Sentence Transformers model
- Writing the processed documents to the document store

The script can be executed directly to load and index all PDFs in a specified directory.
"""

# Connect to WeaviateDocumentStore

# Load the PDF file and create embeddings
def load_and_preprocess_files(directory_path: str):
    document_store = WeaviateDocumentStore(url='http://localhost:8080')

    file_type_router = FileTypeRouter(mime_types=["application/pdf"]) # If other file types are present, add them here
    pdf_converter = PyPDFToDocument()
    document_joiner = DocumentJoiner()
    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="word", split_length=150)
    document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    document_embedder.warm_up()
    document_writer = DocumentWriter(document_store=document_store, duplicate_policy=DuplicatePolicy.OVERWRITE)
    preprocessing_pipeline = Pipeline()
    preprocessing_pipeline.add_component(instance=file_type_router, name="file_type_router")
    preprocessing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
    preprocessing_pipeline.add_component(instance=document_joiner, name="document_joiner")
    preprocessing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
    preprocessing_pipeline.add_component(instance=document_splitter, name="document_splitter")
    preprocessing_pipeline.add_component(instance=document_embedder, name="document_embedder")
    preprocessing_pipeline.add_component(instance=document_writer, name="document_writer")

    preprocessing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
    preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
    preprocessing_pipeline.connect("document_joiner", "document_cleaner")
    preprocessing_pipeline.connect("document_cleaner", "document_splitter")
    preprocessing_pipeline.connect("document_splitter", "document_embedder")
    preprocessing_pipeline.connect("document_embedder", "document_writer")

    preprocessing_pipeline.run({"file_type_router": {"sources": list(Path(directory_path).glob("**/*"))}})
    document_store.client.close()


if __name__ == "__main__": # Checks if file is being run directly
    # Load and index all PDFs in a directory
    directory_path = "documents"  # Update this path
    load_and_preprocess_files(directory_path)
