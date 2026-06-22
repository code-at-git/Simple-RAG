# SimpleRAG

A lightweight, local Retrieval-Augmented Generation (RAG) system built with Python, FAISS, and Ollama. It scans folders for PDF documents, indexes their contents locally using vector embeddings, tracks changes via file hashing, and answers queries using a local LLM.

## Features

- **100% Local**: Keeps data private using local embeddings (`nomic-embed-text`) and models (`llama3`) via Ollama.
- **Incremental Indexing**: Uses MD5 hashing to check files so it only processes new or modified PDFs.
- **Fast Vector Search**: Powered by Facebook AI Similarity Search (FAISS) for quick context retrieval.
- **Clean Token Chunking**: Splits document text cleanly on word boundaries to preserve contextual meaning.

## Prerequisites

1. Install [Ollama](https://ollama.com).
2. Pull the required models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3
   or
   ollama pull quen3:8b
   ```

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com
cd Simple-RAG
pip install numpy faiss-cpu pypdf ollama
```

## Quick Start

Create a folder named `documents` and drop your PDF files into it. Then run the following script:

```python
from rag import SimpleRAG

# Initialize the RAG system
rag = SimpleRAG()

# Ingest PDFs from your directory
rag.add_folder("documents")

# Ask questions about your documents
response = rag.query("What is the main finding in the research paper?")
print("\nResponse:\n", response)
```

## Project Structure

```text
├── rag.py               # Main RAG implementation class
└── rag_data/            # data files for RAG
└── rag_index/           # Created automatically to store state
    ├── index.faiss      # FAISS vector database index
    ├── chunks.pkl       # Saved document chunks mapped to file names
    └── hashes.json      # File MD5 tracking to avoid re-indexing
```
