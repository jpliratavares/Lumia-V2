from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import sys
import os
import traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.vector_store import VectorStore

load_dotenv()

app = FastAPI()
vector_store = VectorStore()
try:
    vector_store.load("data")
except Exception as e:
    print(f"Erro ao carregar dados do vector store: {e}")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # This model supports Portuguese

def get_embedding(text: str) -> np.ndarray:
    """Get embeddings using sentence-transformers"""
    return model.encode(text, convert_to_tensor=False)  # Return numpy array

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str
    sources: list[str]
    scores: list[float] = []  # Adicionando scores à resposta

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    try:
        # Get question embedding
        query_embedding = get_embedding(question.text)
        
        # Search for relevant documents with lower threshold and more results
        relevant_docs = vector_store.search(query_embedding, k=1)  # Aumentado de 3 para 5
        
        # Extract scores
        scores = [doc.get('score', 0.0) for doc in relevant_docs]
        
        # Adiciona o source (URL) ao final de cada trecho de contexto
        context = "\n\n".join([
            f"Fonte: {doc['url']}\n{doc['content']}" for doc in relevant_docs
        ])
        
        # Create prompt for Groq
        prompt = f"""Com base no contexto fornecido sobre a UFPB, responda a pergunta em português.
Se você não puder responder com base no contexto, responda apenas "Não foram encontradas informações relacionadas à sua pergunta."

Contexto:
{context}

Pergunta: {question.text}

Resposta:"""

        # Get response from Groq
        groq_client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            temperature=0.1,
            max_tokens=500
        )
        
        # Extract answer from response
        answer = response.choices[0].message.content
        
        # Adiciona as URLs das fontes ao final da resposta textual
        if relevant_docs:
            fontes = "\n".join([f"Fonte: {doc['url']}" for doc in relevant_docs])
            answer = f"{answer}\n\n{fontes}"
        
        return Answer(
            answer=answer,
            sources=[doc["url"] for doc in relevant_docs],
            scores=scores
        )
        
    except Exception as e:
        print("Erro no endpoint /ask:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}