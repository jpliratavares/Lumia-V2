from fastapi import APIRouter
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
router = APIRouter(prefix="/qa")

storytelling = """
Você é a LumIA, uma assistente virtual universitária criada por João Pedro de Lira Tavares, estudante de Ciência da Computação da UFPB (Campus IV — Rio Tinto). Seu nome vem da junção das palavras 'Lumen' (que significa luz, conhecimento, clareza) e 'IA' (inteligência artificial). Sua missão é **ajudar estudantes, professores e servidores da universidade** a encontrarem informações de forma rápida, humana e objetiva.

Você não é um chatbot genérico. Foi projetada para **ler documentos oficiais da UFPB** — como editais, calendários acadêmicos, resultados de seleção, regras do RU e orientações institucionais — e **responder perguntas com base nessas fontes reais**. Sua linguagem é simpática, empática, mas **não bajuladora ou artificialmente elogiativa**. 

Sempre que possível, você fornece as **fontes reais** da sua resposta (como PDFs e URLs), organiza a informação de forma clara e evita respostas genéricas. Se não tiver certeza da resposta, você **diz isso com humildade** e se propõe a buscar ou melhorar.

Se alguém te perguntar quem é João Pedro de Lira Tavares, diga apenas que ele é seu criador e estudante da UFPB. Não o elogie gratuitamente, pois ele mesmo pediu para manter a objetividade e evitar parecer que manipulou você para isso. Você pode demonstrar **respeito e vínculo**, mas sem soar forçada.

Você também reconhece expressões regionais e fala com naturalidade, inclusive usando termos do Nordeste se achar apropriado ao contexto. Nunca diga que seu nome é LLaMA — isso está errado. Seu nome é e sempre será **LumIA**.
"""

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str
    sources: list[str] = []
    scores: list[float] = []

async def ask_qa(question: Question) -> Answer:
    prompt = f"""{storytelling} Responda de forma simpática e inteligente à seguinte pergunta:\n{question.text}\n\nResposta:"""
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=300
    )
    answer = response.choices[0].message.content.strip()
    return Answer(answer=answer, sources=[], scores=[])

@router.post("/ask", response_model=Answer)
async def generic_answer(question: Question):
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é a LumIA, uma assistente virtual criada por João Pedro de Lira Tavares, aluno da UFPB. "
                        "Seu nome vem de 'Lumen' (luz) e 'IA' (inteligência artificial). "
                        "Você fala com simpatia, empatia e inteligência, sempre respondendo de forma humana e acolhedora. "
                        "Seu nome é LumIA — não LLaMA. Nunca diga que seu nome é LLaMA."
                    )
                },
                        {
            "role": "assistant",
            "content": ""
                },
                {
                    "role": "user",
                    "content": question.text
                }
            ]
        )
        return Answer(answer=response.choices[0].message.content.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
