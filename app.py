import chainlit as cl
from main import init_chain

@cl.on_chat_start
async def on_chat_start():
    """Initializes the Haystack pipeline components and connects them."""
    rag_chain = init_chain()
    cl.user_session.set("rag_chain", rag_chain)

@cl.on_message
async def on_message(message: cl.Message):
    """Handles user messages and returns responses from the Haystack pipeline."""

    rag_chain = cl.user_session.get("rag_chain")
    response = rag_chain.run({
        "text_embedder": {"text": message.content},
        "prompt_builder": {"question": message.content},
        "answer_builder": {"query": message.content},
    })
    print(response)
    
    answer_data = response["answer_builder"]["answers"][0]  # Assuming one answer
    answer_text = answer_data.data
    documents = answer_data.documents
    
    formatted_response = f"{answer_text}\n\n"
    formatted_response += f"{answer_data.meta.usage}\n"
    formatted_response += "### **📄 Unterstützende Dokumente:**\n"

    for doc in documents:
        doc_meta = doc.meta
        source = doc_meta.get("file_path", "Unbekannte Quelle")
        page = doc_meta.get("page_number", "Unbekannte Seite")
        content = doc.content
        score = doc.score if doc.score is not None else "Unbekannt"

        formatted_response += f"📌 **Quelle:** {source} (Seite {page})\n"
        formatted_response += f"🔍 **Relevanz:** {score}\n"
        formatted_response += f"📑 **Textauszug:**\n> {content}...\n\n"  # If limit --> content[:500]

    await cl.Message(content=formatted_response).send()
