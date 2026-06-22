import os
import json
import hashlib
import pickle
import numpy as np
import faiss
from pypdf import PdfReader
import ollama

class SimpleRAG:
    def __init__(self, index_dir="rag_index"):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)

        self.index_file = os.path.join(index_dir, "index.faiss")
        self.chunks_file = os.path.join(index_dir, "chunks.pkl")
        self.hash_file = os.path.join(index_dir, "hashes.json")

        self.hashes = self._load_hashes()
        self.chunks = []

        self.index = None

        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            self.chunks = pickle.load(open(self.chunks_file, "rb"))

    def _load_hashes(self):
        if os.path.exists(self.hash_file):
            with open(self.hash_file, "r") as f:
                return json.load(f)
        return {}

    def _save_state(self):
        faiss.write_index(self.index, self.index_file)
        pickle.dump(self.chunks, open(self.chunks_file, "wb"))
        with open(self.hash_file, "w") as f:
            json.dump(self.hashes, f)

    def _file_hash(self, path):
        h = hashlib.md5()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    def _load_pdf(self, path):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _chunk(self, text, size=1000, overlap=150):
        chunks = []
        i = 0
        while i < len(text):
            chunks.append(text[i:i + size])
            i += size - overlap
        return chunks

    def _embed(self, text):
        res = ollama.embeddings(
            model="nomic-embed-text",
            prompt=text
        )
        return np.array(res["embedding"], dtype=np.float32)

    def add_folder(self, folder):
        for file in os.listdir(folder):
            if not file.endswith(".pdf"):
                continue

            path = os.path.join(folder, file)
            file_hash = self._file_hash(path)

            if self.hashes.get(path) == file_hash:
                continue

            print(f"Indexing: {file}")

            text = self._load_pdf(path)
            chunks = self._chunk(text)

            embeddings = np.array([self._embed(c) for c in chunks], dtype=np.float32)

            if self.index is None:
                dim = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dim)

            self.index.add(embeddings)
            self.chunks.extend(chunks)

            self.hashes[path] = file_hash

        self._save_state()

    def search(self, query, k=4):
        q = self._embed(query).reshape(1, -1)
        _, idx = self.index.search(q, k)
        return [self.chunks[i] for i in idx[0]]

    def ask(self, question):
        context = "\n\n".join(self.search(question))

        prompt = f"""
You are an assistant.

Context:
{context}

Question:
{question}

Answer clearly and accurately.
"""

        res = ollama.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": prompt}]
        )

        return res["message"]["content"]


if __name__ == "__main__":
    rag = SimpleRAG()

    rag.add_folder("rag_data")

    print(rag.ask("What is the vacation policy?"))
