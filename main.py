from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.weaviate import WeaviateEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.readers import ExtractiveReader
from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from haystack import Pipeline
from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore
from haystack.utils.hf import HFGenerationAPIType
import os
import logging
from dotenv import load_dotenv

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

load_dotenv()

if "HF_API_TOKEN" not in os.environ:
    raise EnvironmentError("Hugging Face token not found in environment variables.")

def queryDB(question: str): 
    document_store = WeaviateDocumentStore(url='http://localhost:8080')
    chat_template = [
        ChatMessage.from_system(
            """
            You are a helpful, respectful and honest assistant. 
            Please answer only in german. 
            Use concise and clear language. 
            Please use gender neutral language or gender with "*".
            """),
        ChatMessage.from_user(
            """
            Given the following information, answer the question.

            Context:
            {% for document in documents %}
                {{ document.content }}
            {% endfor %}

            Question: {{question}}
            Answer:
            """
        )
    ]

    embedder =  SentenceTransformersTextEmbedder(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    retriever = WeaviateEmbeddingRetriever(document_store=document_store, top_k=5)
    reader = ExtractiveReader()
    reader.warm_up()

    prompt_builder = ChatPromptBuilder(template=chat_template)
    llm = HuggingFaceAPIChatGenerator(
        api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API, 
        api_params={"model": "microsoft/Phi-3.5-mini-instruct"},
        generation_kwargs={"max_new_tokens": 512}
        )

    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", embedder)
    query_pipeline.add_component("retriever", retriever)
    query_pipeline.add_component("reader", reader)
    query_pipeline.add_component("prompt_builder", prompt_builder)
    query_pipeline.add_component("llm", llm)

    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    query_pipeline.connect("retriever.documents", "reader.documents")
    query_pipeline.connect("reader.answers", "prompt_builder.documents")
    query_pipeline.connect("prompt_builder.prompt", "llm.messages")

    response = query_pipeline.run({
            "text_embedder": {"text": question},
            "reader": {"query": question},
            "prompt_builder": {"question": question}
        }, include_outputs_from={"reader", "llm"})
    document_store.client.close()
    return response["llm"]

if __name__ == "__main__": 
    queryDB()
