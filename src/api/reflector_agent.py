# src/agents/reflector_agent.py
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

ROUTING_PROMPT_TEMPLATE = """
Usuário: "{question}"

Você é um roteador inteligente. Esta pergunta deve ser enviada para:
- QA -> se for sobre identidade da assistente, dúvidas gerais, interação humana, curiosidades, etc.
- QB -> se for sobre informações que dependem de documentos indexados.
- COLLAB -> se precisar de interpretação do QA e depois busca com o QB.
- TESTE -> se for uma pergunta que pede a geração de perguntas baseadas nos documentos, para testar a base.

Responda apenas com UMA dessas opções: QA / QB / COLLAB / TESTE.
"""

def decidir_fluxo(pergunta: str) -> str:
    prompt = ROUTING_PROMPT_TEMPLATE.format(question=pergunta)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        temperature=0,
        max_tokens=5,
        messages=[{"role": "user", "content": prompt}]
    )
    resposta = response.choices[0].message.content.strip().upper()
    return resposta if resposta in ["QA", "QB", "COLLAB", "TESTE"] else "QA"
