import numpy as np
import json
import os
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# --- Agente de Indexação ---
class IndexAgent:
    def __init__(self):
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents: List[Dict[str, str]], embeddings: np.ndarray):
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents must match number of embeddings")
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.documents.extend(documents)

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "documents.json"), "w", encoding="utf-8") as f:
            json.dump(self.documents, f)
        if self.embeddings is not None:
            np.save(os.path.join(directory, "embeddings.npy"), self.embeddings)

    def load(self, directory: str):
        with open(os.path.join(directory, "documents.json"), "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        embeddings_path = os.path.join(directory, "embeddings.npy")
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)

# --- Agente de Busca ---
class SearchAgent:
    def __init__(self, index_agent: IndexAgent, similarity_threshold: float = 0.4):
        self.index_agent = index_agent
        self.similarity_threshold = similarity_threshold

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        if self.index_agent.embeddings is None or len(self.index_agent.documents) == 0:
            return []
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.index_agent.embeddings)[0]
        valid_indices = np.where(similarities >= self.similarity_threshold)[0]
        if len(valid_indices) == 0:
            return []
        valid_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
        valid_indices = valid_indices[:k]
        results = []
        for idx in valid_indices:
            result = self.index_agent.documents[idx].copy()
            result['score'] = float(similarities[idx])
            results.append(result)
        return results

# --- Orchestrator ---
class VectorStoreOrchestrator:
    def __init__(self, similarity_threshold: float = 0.4):
        self.index_agent = IndexAgent()
        self.search_agent = SearchAgent(self.index_agent, similarity_threshold)

    def add_documents(self, documents: List[Dict[str, str]], embeddings: np.ndarray):
        self.index_agent.add_documents(documents, embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        return self.search_agent.search(query_embedding, k)

    def save(self, directory: str):
        self.index_agent.save(directory)

    def load(self, directory: str):
        self.index_agent.load(directory)

# Para compatibilidade retroativa
VectorStore = VectorStoreOrchestrator

class FaissIndexAgent:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.documents = []
        self.index = None
        self._load()

    def _load(self):
        docs_path = os.path.join(self.data_dir, "documents.json")
        faiss_path = os.path.join(self.data_dir, "faiss.index")
        if os.path.exists(docs_path):
            with open(docs_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
        if os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)
        else:
            self.index = faiss.IndexFlatL2(384)

    def search(self, query_embedding: np.ndarray, k=5):
        if self.index is None or len(self.documents) == 0:
            return []
        D, I = self.index.search(query_embedding.reshape(1, -1), k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx].copy()
            doc['score'] = float(-dist)  # Negativo porque L2, para parecer score
            results.append(doc)
        return results

# --- Faiss Vector Store Orchestrator ---
class FaissVectorStore:
    def __init__(self, data_dir="data"):
        self.agent = FaissIndexAgent(data_dir)

    def search(self, query_embedding: np.ndarray, k=5):
        return self.agent.search(query_embedding, k)