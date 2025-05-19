from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from groq import Groq
import os
from api.qb_agent import ask_qb
from api.qa_endpoint import ask_qa

router = APIRouter()

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str
    sources: list[str]
    scores: list[float]

def classificar_pergunta(pergunta: str) -> str:
    prompt = f"""
Você é um classificador de intenções. Receba uma pergunta de um usuário universitário e responda apenas com 'qa' ou 'qb'.

- 'qa' = pergunta geral ou pessoal (nome da assistente, piadas, idade, ajuda etc.)
- 'qb' = pergunta sobre documentos, editais, resultados, PDFs, dados acadêmicos.

Pergunta: "{pergunta}"
Classificação:
""".strip()

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
        temperature=0,
        max_tokens=3
    )
    resposta = response.choices[0].message.content.strip().lower()
    return "qa" if "qa" in resposta else "qb"

@router.post("/ask", response_model=Answer)
async def ask(question: Question, threshold: float = Query(0.4, description="Threshold para busca QB")):
    try:
        destino = classificar_pergunta(question.text)
        if destino == "qa":
            return await ask_qa(question)
        else:
            return await ask_qb(question, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
