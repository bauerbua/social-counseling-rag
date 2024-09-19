# chainlit_mistral_nonlocal

Simple scripts to deploy a complete LLM application, using ChainLit and pretrained Mistral model (non-local).

The assumption is that the script is running on localhost but the porting to AWS is quite straightforward

Depending on the script I am using either just Haystack to orchestrate the RAG but only temporarily, or the full vector storing using also Weaviate.
The old script using Bootstrap.js has been replaced by Chainlit, but the architecture is the same:

![Architecture](https://github.com/alecrimi/chainlit_mistral_nonlocal/blob/main/architectureLLM.png?raw=true "Architecture")
 

