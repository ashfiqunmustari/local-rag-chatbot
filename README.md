# Local RAG Chatbot

A local AI chatbot that can answer questions from personal documents using Retrieval-Augmented Generation (RAG)

## Features

- Vector search with FAISS
- Chunked document indexing
- Conversational memory
- History-aware query reformulation
- Local LLaMA model (via Ollama)
- Works with any text content
- Multi-turn chat loop


## Overview

1. **Text Splitting** — chunks + overlap preserve context  
2. **Embeddings** — with Sentence Transformers  
3. **Vector Store** — FAISS for similarity search  
4. **Retriever**  
   - history-aware
   - reformulates vague q uestions  
5. **RAG Chain**  
   - retrieve top chunks  
   - generate grounded answer  
6. **Memory** — keeps chat context  
7. **Local LLM** — served by Ollama


## Tech Stack

- Python
- LangChain
- FAISS
- HuggingFace Embeddings
- Ollama
- Local LLaMA 3 model
