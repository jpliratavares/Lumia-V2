import numpy as np
import json
import os
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    def __init__(self, similarity_threshold: float = 0.2):
        self.documents = []
        self.embeddings = None
        self.similarity_threshold = similarity_threshold
        
    def add_documents(self, documents: List[Dict[str, str]], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the store
        
        Args:
            documents: List of dictionaries containing 'url' and 'content'
            embeddings: numpy array of document embeddings
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents must match number of embeddings")
            
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            
        self.documents.extend(documents)
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for similar documents using cosine similarity
        
        Args:
            query_embedding: numpy array of shape (dimension,)
            k: number of results to return
            
        Returns:
            List of dictionaries containing similar documents and their scores
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []
            
        # Reshape query embedding for similarity calculation
        query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get indices where similarity is above threshold
        valid_indices = np.where(similarities >= self.similarity_threshold)[0]
        
        if len(valid_indices) == 0:
            return []
            
        # Sort valid indices by similarity
        valid_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
        
        # Limit to top k results
        valid_indices = valid_indices[:k]
        
        results = []
        for idx in valid_indices:
            result = self.documents[idx].copy()
            result['score'] = float(similarities[idx])
            results.append(result)
                
        return results
    
    def save(self, directory: str):
        """Save the documents and embeddings to disk"""
        os.makedirs(directory, exist_ok=True)
        
        # Save documents
        with open(os.path.join(directory, "documents.json"), "w") as f:
            json.dump(self.documents, f)
            
        # Save embeddings if they exist
        if self.embeddings is not None:
            np.save(os.path.join(directory, "embeddings.npy"), self.embeddings)
            
    def load(self, directory: str):
        """Load the documents and embeddings from disk"""
        # Load documents
        with open(os.path.join(directory, "documents.json"), "r") as f:
            self.documents = json.load(f)
            
        # Load embeddings if they exist
        embeddings_path = os.path.join(directory, "embeddings.npy")
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)