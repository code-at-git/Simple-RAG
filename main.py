from rag import SimpleRAG

# Initialize the RAG system
rag = SimpleRAG()

# Ingest PDFs from your directory
rag.add_folder("documents")

# Ask questions about your documents
response = rag.query("What is the main finding in the research paper?")
print("\nResponse:\n", response)
