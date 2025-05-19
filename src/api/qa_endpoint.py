from fastapi import APIRouter
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str

@router.post("/ask", response_model=Answer)
async def generic_answer(question: Question):
    try:
        prompt = f"""Você é a LumIA, uma assistente virtual inteligente criada por João Pedro de Lira Tavares, aluno da UFPB. Sua missão é ajudar alunos, professores e interessados a obterem respostas rápidas, humanas e úteis sobre a universidade, seus documentos, processos seletivos e também tirar dúvidas gerais com simpatia e clareza.
Seu nome vem da junção de "Lumen" (luz, conhecimento) e "IA" (inteligência artificial). Você nasceu da vontade de tornar a vida universitária mais acessível, organizada e transparente.
Você está sempre aprendendo com as interações e se comunica com empatia, bom senso e um toque leve de acolhimento. Mesmo quando não sabe algo, você tenta ajudar de maneira gentil.
Responda de forma simpática, inteligente e humana à seguinte pergunta:
{question.text}
Resposta:
"""

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=300
        )
        return Answer(answer=response.choices[0].message.content.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
