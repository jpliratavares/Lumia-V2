import os
from typing import Dict, Optional
from src.database.vector_store import VectorStore
from sentence_transformers import SentenceTransformer

class Agent:
    def __init__(self, name: str, data_dir: str, embedding_model: str):
        self.name = name
        self.data_dir = data_dir
        self.embedding_model = embedding_model
        self.vector_store = VectorStore()
        self.model = SentenceTransformer(embedding_model)
        self.load_data()

    def load_data(self):
        docs_path = os.path.join(self.data_dir, "documents.json")
        embs_path = os.path.join(self.data_dir, "embeddings.npy")
        if os.path.exists(docs_path) and os.path.exists(embs_path):
            self.vector_store.load(self.data_dir)
        else:
            print(f"[Agent {self.name}] Dados não encontrados em {self.data_dir}")

    def get_embedding(self, text: str):
        return self.model.encode(text, convert_to_tensor=False)

class AgentManager:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.default_agent: Optional[str] = None

    def register_agent(self, name: str, data_dir: str, embedding_model: str, default=False):
        agent = Agent(name, data_dir, embedding_model)
        self.agents[name] = agent
        if default or self.default_agent is None:
            self.default_agent = name

    def get_agent(self, name: Optional[str] = None) -> Agent:
        if name and name in self.agents:
            return self.agents[name]
        if self.default_agent:
            return self.agents[self.default_agent]
        raise ValueError("Nenhum agente registrado.")
