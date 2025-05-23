# src/api/ask_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal
from api.reflector_agent import decidir_fluxo
from api.qa_endpoint import ask_qa
from api.qb_agent import ask_qb

router = APIRouter()

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str
    sources: list[str] = []
    scores: list[float] = []

@router.post("/ask", response_model=Answer)
async def ask_router(question: Question):
    try:
        fluxo = decidir_fluxo(question.text)

        if fluxo == "QA":
            return await ask_qa(question)
        elif fluxo == "QB":
            return await ask_qb(question)
        elif fluxo == "COLLAB":
            interpretacao = await ask_qa(question)
            print(f"[COLLAB] Pergunta gerada pelo QA: {interpretacao.answer}")
            resposta_final = await ask_qb(Question(text=interpretacao.answer))
            print(f"[COLLAB] Resposta final do QB: {resposta_final.answer}")
            return resposta_final

        elif fluxo == "TESTE":
            # Gera perguntas baseadas na base indexada
            return await ask_qa(Question(text=f"Gere perguntas de exemplo com base nos documentos disponíveis. {question.text}"))

        return await ask_qa(question)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
