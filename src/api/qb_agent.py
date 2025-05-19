from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from agents.agent_manager import AgentManager
from groq import Groq
from dotenv import load_dotenv
import os

router = APIRouter(prefix="/qb")

load_dotenv()

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str
    sources: list[str]
    scores: list[float] = []

# Inicializa apenas com o agente qb
agent_manager = AgentManager()
agent_manager.register_agent(
    name="qb",
    data_dir="data/",
    embedding_model='paraphrase-multilingual-MiniLM-L12-v2',
    default=True
)

@router.post("/ask", response_model=Answer)
async def ask_qb(
    question: Question,
    threshold: float = Query(0.4, description="Threshold de similaridade para busca de documentos")
):
    try:
        ag = agent_manager.get_agent("qb")
        query_embedding = ag.get_embedding(question.text)
        relevant_docs = ag.vector_store.search(query_embedding, k=8)

        sources = list(set(doc["url"].split('#')[0] for doc in relevant_docs))
        context = "\n\n".join([
    f"Fonte: {doc['url']}\n{doc['content'][:1000]}"  # 1000 caracteres ≈ 250 tokens por doc
    for doc in relevant_docs
])
        scores = [doc.get("score", 0.0) for doc in relevant_docs]

        if not relevant_docs:
            return Answer(answer="Nenhum documento relevante encontrado.", sources=[], scores=[])

        prompt = f"""Com base no contexto abaixo, responda a pergunta em português.\nSe não houver contexto suficiente, diga isso claramente.\n\nContexto:\n{context}\n\nPergunta: {question.text}\n\nResposta:"""
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.1,
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()
        return Answer(answer=answer, sources=sources, scores=scores)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documentos")
async def list_documents_qb():
    ag = agent_manager.get_agent("qb")
    docs = ag.vector_store.agent.documents if hasattr(ag.vector_store, 'agent') else ag.vector_store.index_agent.documents
    valid_docs = [doc for doc in docs if isinstance(doc, dict) and "url" in doc and "content" in doc]
    if not valid_docs:
        raise HTTPException(status_code=404, detail="Nenhum documento encontrado para o agente QB.")
    return [
        {"url": doc["url"], "preview": doc["content"][:200]} for doc in valid_docs
    ]

@router.get("/health")
async def health_check_qb():
    return {"status": "healthy"}

async def ask_qb_internal(question: Question, threshold: float = 0.4) -> Answer:
    ag = agent_manager.get_agent("qb")
    query_embedding = ag.get_embedding(question.text)
    relevant_docs = ag.vector_store.search(query_embedding, k=8)

    if not relevant_docs:
        return Answer(answer="Nenhum documento relevante encontrado.", sources=[], scores=[])

    sources = list(set(doc["url"].split('#')[0] for doc in relevant_docs))
    context = "\n\n".join([f"Fonte: {doc['url']}\n{doc['content']}" for doc in relevant_docs])
    scores = [doc.get("score", 0.0) for doc in relevant_docs]

    prompt = f"""Com base no contexto abaixo, responda a pergunta em português.\nSe não houver contexto suficiente, diga isso claramente.\n\nContexto:\n{context}\n\nPergunta: {question.text}\n\nResposta:"""
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
        temperature=0.1,
        max_tokens=500
    )
    answer = response.choices[0].message.content.strip()
    return Answer(answer=answer, sources=sources, scores=scores)

