from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.weaviate import WeaviateEmbeddingRetriever


document_store = WeaviateDocumentStore(url='http://localhost:8080')

query_pipeline = Pipeline()
embedder =  SentenceTransformersTextEmbedder(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
retriever = WeaviateEmbeddingRetriever(document_store=document_store)

query_pipeline.add_component("text_embedder", embedder)
query_pipeline.add_component("retriever", retriever)
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

query = "Was ist das Ziel des Behinderten-Gleichstellungspakets?"

result = query_pipeline.run({"text_embedder": {"text": query}})

print(result["retriever"]["documents"][0])
document_store.client.close()