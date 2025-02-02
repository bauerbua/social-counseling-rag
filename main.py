from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.weaviate import WeaviateEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.builders.answer_builder import AnswerBuilder
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


def init_chain(): 
    document_store = WeaviateDocumentStore(url='http://localhost:8080')
    chat_template = [
        ChatMessage.from_system(
            """
            <|system|>
            You are a helpful, respectful, and honest assistant. 
            Follow these strict instructions:  

            - Answer **ONLY in German**.  
            - Use **gender-neutral language** or **gender with '*'**.<|end|>
            """),
        ChatMessage.from_user(
            """
            <|user|>
            Given the following information, answer the question.

            ### Kontext:  
            {% for document in documents %}
            📌 **Quelle:** {{ document.meta["file_path"] if document.meta["file_path"] else "Unbekannte Quelle" }}  
            🔗 **Relevanz:** {{ document.score if document.score else "Unbekannt" }}  
            📝 **Text:**  
            {{ document.content }}  
            ---  
            {% endfor %}  

            ### Frage:  
            {{ question }}<|end|>
            
            <|assistant|>
            """
        )
    ]

    embedder =  SentenceTransformersTextEmbedder(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    retriever = WeaviateEmbeddingRetriever(document_store=document_store, top_k=3)
    reader = ExtractiveReader()
    reader.warm_up()

    prompt_builder = ChatPromptBuilder(template=chat_template)
    llm = HuggingFaceAPIChatGenerator(
        api_type=HFGenerationAPIType.SERVERLESS_INFERENCE_API, 
        api_params={"model": "microsoft/Phi-3.5-mini-instruct", "max_new_tokens": 512},
        stop_words=["<|endoftext|>"],
        )

    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", embedder)
    query_pipeline.add_component("retriever", retriever)
    query_pipeline.add_component("prompt_builder", prompt_builder)
    query_pipeline.add_component("llm", llm)
    query_pipeline.add_component("answer_builder", AnswerBuilder())

    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    query_pipeline.connect("retriever.documents", "prompt_builder.documents")
    query_pipeline.connect("prompt_builder.prompt", "llm.messages")
    query_pipeline.connect("llm.replies", "answer_builder.replies")
    query_pipeline.connect("retriever", "answer_builder.documents")
    
    return query_pipeline

def queryDB(question: str): 
    query_pipeline = init_chain()
    response = query_pipeline.run({
            "text_embedder": {"text": question},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        })
    return response

if __name__ == "__main__": 
    # question = "Was ist das Ziel des Behinderten-Gleichstellungspakets?"
    # question = "Wann liegt eine unmittelbare Diskriminierung vor?"
    question = "Wer ist durch das Behinderten-Einstellungsgesetz vor Diskriminierung geschützt?"
    result = queryDB(question)
    print(result)
