from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from groq import Groq
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import sys
import os
import traceback
from typing import Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.vector_store import VectorStore
from cardapio_manager import consultar_cardapio
from src.agents.agent_manager import AgentManager

load_dotenv()

app = FastAPI()

# Inicializa o sistema de agentes
agent_manager = AgentManager()
# Registra o agente padrão (UFPB)
agent_manager.register_agent(
    name="ufpb",
    data_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"),
    embedding_model='paraphrase-multilingual-MiniLM-L12-v2',
    default=True
)

try:
    # Carrega dados de data se existir, senão de data
    if os.path.exists(os.path.join("data", "documents.json")) and os.path.exists(os.path.join("data", "embeddings.npy")):
        vector_store.load("data")
        print("[QA] Dados carregados de data/")
    else:
        vector_store.load("data")
        print("[QA] Dados carregados de data/")
    
    # <<< DIAGNÓSTICO ADICIONAL >>>
    num_docs = len(vector_store.index_agent.documents)
    if vector_store.index_agent.embeddings is not None:
        num_embeddings = vector_store.index_agent.embeddings.shape[0]
    else:
        num_embeddings = 0
    print(f"[DIAGNÓSTICO] Documentos carregados: {num_docs}")
    print(f"[DIAGNÓSTICO] Embeddings carregados: {num_embeddings}")

except Exception as e:
    print(f"Erro ao carregar dados do vector store: {e}")

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str
    sources: list[str]
    scores: list[float] = []  # Adicionando scores à resposta

class CardapioResponse(BaseModel):
    data: str
    refeicao: str
    campus: str
    salada: str
    prato_principal: str
    acompanhamentos: str
    bebidas: str

# Dicionário simples de regionalismos (exemplo inicial)
REGIONALISMOS = {
    # Norte
    "arre diacho": "norte (Acre)",
    "arrodear": "norte (Acre)",
    "espocar": "norte (Acre)",
    "xiringar": "norte (Acre)",
    "extrato": "norte (Acre)",
    "de bubuia": "norte (Amazonas)",
    "papoco": "norte (Amazonas)",
    "até o tucupi": "norte (Pará)",
    "égua": "norte (Pará)",
    "égua de largura": "norte (Pará)",
    "pai d’égua": "norte (Pará)",
    "pavulagem": "norte (Pará)",
    "carapanã": "norte (Pará)",
    "pomba lesa": "norte (Amapá)",
    "pegar o beco": "norte (Amazonas)",
    "de rocha": "norte (Pará)",
    "capar o gato": "norte (Pará)",
    "marrapaz": "norte (Rondônia)",
    "leseira": "norte (Rondônia)",
    "curumim": "norte (Roraima)",
    "cunhantã": "norte (Amazonas)",
    "brocado": "norte (Roraima)",
    "levou o farelo": "norte (Pará)",

    # Nordeste
    "abestado": "nordeste (Ceará)",
    "abilolado": "nordeste (Paraíba)",
    "aluado": "nordeste (Ceará)",
    "aperreado": "nordeste (Ceará)",
    "armaria": "nordeste (Pernambuco)",
    "arretado": "nordeste (Pernambuco)",

    # Norte
    "arre diacho": "norte (Acre)",
    "arrodear": "norte (Acre)",
    "espocar": "norte (Acre)",
    "xiringar": "norte (Acre)",
    "extrato": "norte (Acre)",
    "de bubuia": "norte (Amazonas)",
    "papoco": "norte (Amazonas)",
    "até o tucupi": "norte (Pará)",
    "égua": "norte (Pará)",
    "égua de largura": "norte (Pará)",
    "pai d’égua": "norte (Pará)",
    "pavulagem": "norte (Pará)",
    "carapanã": "norte (Pará)",
    "pomba lesa": "norte (Amapá)",
    "pegar o beco": "norte (Amazonas)",
    "de rocha": "norte (Pará)",
    "capar o gato": "norte (Pará)",
    "marrapaz": "norte (Rondônia)",
    "leseira": "norte (Rondônia)",
    "curumim": "norte (Roraima)",
    "cunhantã": "norte (Amazonas)",
    "brocado": "norte (Roraima)",
    "levou o farelo": "norte (Pará)",

    # Nordeste
    "abestado": "nordeste (Ceará)",
    "abilolado": "nordeste (Paraíba)",
    "aluado": "nordeste (Ceará)",
    "aperreado": "nordeste (Ceará)",
    "armaria": "nordeste (Pernambuco)",
    "arretado": "nordeste (Pernambuco)",
    "azougado": "nordeste (Alagoas)",
    "avalie": "nordeste (Alagoas)",
    "avexado": "nordeste (Pernambuco)",
    "avoar": "nordeste (Paraíba)",
    "bagaceira": "nordeste (Pernambuco)",
    "baixa da égua": "nordeste (Ceará)",
    "basculho": "nordeste (Pernambuco)",
    "bexiga taboca": "nordeste (Rio Grande do Norte)",
    "budejar": "nordeste (Piauí)",
    "buchuda": "nordeste (Pernambuco)",
    "buliçoso": "nordeste (Pernambuco)",
    "cabra da peste": "nordeste (Nordeste)",
    "caba de pêia": "nordeste (Alagoas)",
    "cafuringa": "nordeste (Pernambuco)",
    "desmantelar": "nordeste (Ceará)",
    "diabéisso": "nordeste (Ceará)",
    "estrambólico": "nordeste (Pernambuco)",
    "estribado": "nordeste (Ceará)",
    "fuleiro": "nordeste (Ceará)",
    "fumbambento": "nordeste (Sergipe)",
    "fuzuê": "nordeste (Bahia)",
    "galado": "nordeste (Rio Grande do Norte)",
    "galego": "nordeste (Paraíba)",
    "gastura": "nordeste (Ceará)",
    "invocado": "nordeste (Maranhão)",
    "jerimum": "nordeste (Pernambuco)",
    "lapada": "nordeste (Pernambuco)",
    "macaxeira": "nordeste (Pernambuco)",
    "macambúzio": "nordeste (Pernambuco)",
    "mainha": "nordeste (Bahia)",
    "painho": "nordeste (Bahia)",
    "mangar": "nordeste (Ceará)",
    "migué": "nordeste (Bahia)",
    "miolo de pote": "nordeste (Paraíba)",
    "oxente": "nordeste (Bahia)",
    "oxe": "nordeste (Bahia)",
    "pantim": "nordeste (Pernambuco)",
    "ruma": "nordeste (Ceará)",
    "sustança": "nordeste (Pernambuco)",
    "tabacudo": "nordeste (Pernambuco)",
    "té doido": "nordeste (Maranhão)",
    "triscar": "nordeste (Piauí)",
    "visse": "nordeste (Pernambuco)",
    "vixe": "nordeste (Bahia)",
    "vôti": "nordeste (Sergipe)",
    "zuada": "nordeste (Pernambuco)",

    # Centro-Oeste
    "abiscoitar": "centro-oeste (Goiás)",
    "anêim": "centro-oeste (Goiás)",
    "arruinou": "centro-oeste (Goiás)",
    "bão demais da conta": "centro-oeste (Goiás)",
    "bereré": "centro-oeste (Goiás)",
    "bitelo": "centro-oeste (Mato Grosso)",
    "bocó de fivela": "centro-oeste (Mato Grosso)",
    "cabuloso": "centro-oeste (Distrito Federal)",
    "camelo": "centro-oeste (Distrito Federal)",
    "descabriado": "centro-oeste (Goiás)",
    "empatar": "centro-oeste (Goiás)",
    "goma": "centro-oeste (Mato Grosso do Sul)",
    "lombra": "centro-oeste (Distrito Federal)",
    "mocorongo": "centro-oeste (Goiás)",
    "morgar": "centro-oeste (Distrito Federal)",
    "não dá conta": "centro-oeste (Goiás)",
    "pagar vexa": "centro-oeste (Distrito Federal)",
    "paia": "centro-oeste (Goiás)",
    "pior": "centro-oeste (Mato Grosso do Sul)",
    "rensga": "centro-oeste (Goiás)",
    "tem base?": "centro-oeste (Goiás)",
    "dar um pião": "centro-oeste (Mato Grosso do Sul)",

    # Sudeste
    "B.O.": "sudeste (São Paulo)",
    "capiau": "sudeste (São Paulo)",
    "daora": "sudeste (São Paulo)",
    "de boa": "sudeste (São Paulo)",
    "dois palitos": "sudeste (São Paulo)",
    "jacú": "sudeste (São Paulo)",
    "larica": "sudeste (Rio de Janeiro)",
    "mano": "sudeste (São Paulo)",
    "maneiro": "sudeste (Rio de Janeiro)",
    "marombado": "sudeste (São Paulo)",
    "meu": "sudeste (São Paulo)",
    "meter o pé": "sudeste (Rio de Janeiro)",
    "mermão": "sudeste (Rio de Janeiro)",
    "padoca": "sudeste (São Paulo)",
    "palha": "sudeste (Espírito Santo)",
    "parada": "sudeste (Rio de Janeiro)",
    "pão de sal": "sudeste (Minas Gerais)",
    "picar a mula": "sudeste (Minas Gerais)",
    "bolado": "sudeste (Rio de Janeiro)",
    "irado": "sudeste (Rio de Janeiro)",
    "já é": "sudeste (Rio de Janeiro)",
    "quebrado": "sudeste (São Paulo)",
    "rolê": "sudeste (São Paulo)",
    "sangue bom": "sudeste (São Paulo)",
    "sinistro": "sudeste (Rio de Janeiro)",
    "sô": "sudeste (Minas Gerais)",
    "sussa": "sudeste (São Paulo)",
    "treta": "sudeste (São Paulo)",
    "trem": "sudeste (Minas Gerais)",
    "uai": "sudeste (Minas Gerais)",
    "fragar": "sudeste (Minas Gerais)",
    "bololô": "sudeste (Minas Gerais)",
    "é fria": "sudeste (Rio de Janeiro)",
    "só o pó": "sudeste (Minas Gerais)",
    "nu": "sudeste (Minas Gerais)",

    # Sul
    "bah": "sul (Rio Grande do Sul)",
    "barbaridade": "sul (Rio Grande do Sul)",
    "tchê": "sul (Rio Grande do Sul)",
    "capaz": "sul (Rio Grande do Sul)",
    "tri": "sul (Rio Grande do Sul)",
    "arrecém": "sul (Rio Grande do Sul)",
    "guri": "sul (Rio Grande do Sul)",
    "guria": "sul (Rio Grande do Sul)",
    "piá": "sul (Paraná)",
    "cacetinho": "sul (Rio Grande do Sul)",
    "bergamota": "sul (Rio Grande do Sul)",
    "lancheria": "sul (Rio Grande do Sul)",
    "cusco": "sul (Rio Grande do Sul)",
    "guaipeca": "sul (Rio Grande do Sul)",
    "lagartear": "sul (Santa Catarina)",
    "manezinho": "sul (Santa Catarina)",
    "Tas tolo?": "sul (Santa Catarina)",
    "botar tento": "sul (Santa Catarina)",
    "solito": "sul (Rio Grande do Sul)",
    "atucanado": "sul (Rio Grande do Sul)",
    "esgualepado": "sul (Rio Grande do Sul)",
    "embretrar-se": "sul (Rio Grande do Sul)",
    "pila": "sul (Rio Grande do Sul)",
    "macanudo": "sul (Rio Grande do Sul)",
    "borracho": "sul (Rio Grande do Sul)",
    "baita": "sul (Rio Grande do Sul)",
    "penal": "sul (Paraná)",
    "posar": "sul (Paraná)",
    "gasosa": "sul (Paraná)",
    "vina": "sul (Paraná)",
    "joça": "sul (Paraná)",
    "cozido": "sul (Paraná)",
    "trova": "sul (Rio Grande do Sul)",
    "peleia": "sul (Rio Grande do Sul)",
    "bucha": "sul (Rio Grande do Sul)"
}  

def detectar_regionalismo(texto):
    """Detecta o primeiro regionalismo encontrado no texto e retorna a região."""
    texto_baixo = texto.lower()
    for palavra, regiao in REGIONALISMOS.items():
        if palavra in texto_baixo:
            return regiao, palavra
    return None, None

def is_index_query(text: str) -> Optional[str]:
    """Detecta se a pergunta é sobre os arquivos/documentos indexados."""
    text_lower = text.lower()
    if any(p in text_lower for p in [
        "quais arquivos você tem", "quais documentos você tem", "o que está indexado", "quais pdfs você tem",
        "quais arquivos estão indexados", "quais documentos estão indexados", "quais pdfs estão indexados",
        "lista de arquivos", "lista de documentos", "lista de pdfs", "documentos disponíveis", "pdfs disponíveis"
    ]):
        if "pdf" in text_lower:
            return "pdf"
        return "all"
    return None

def extract_keyword_from_question(text: str) -> str:
    """
    Extrai uma possível palavra-chave para filtro de URL a partir da pergunta do usuário.
    Exemplo: se a pergunta contém 'convocacao-03-2024' ou 'convocação 03/2024', retorna 'convocacao-03-2024'.
    """
    import re
    # Normaliza para minúsculo e sem acentos
    import unicodedata
    def normalize(s):
        return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('ASCII').lower()
    text_norm = normalize(text)
    # Procura padrões tipo 'convocacao', 'comissao', '2024', '03/2024', etc
    # Extrai palavras-chave compostas e datas
    match = re.search(r'(convocacao|comissao|edital|relatorio|comissao de heteroidentificacao)[\w\-/ ]*\d{2,4}', text_norm)
    if match:
        # Remove espaços e barras para formar um slug
        return re.sub(r'[^a-z0-9-]', '-', match.group(0)).replace('--', '-').strip('-')
    # Alternativamente, extrai a maior palavra relevante
    tokens = [t for t in re.split(r'\W+', text_norm) if len(t) > 4]
    if tokens:
        return tokens[0]
    return ""

def split_into_chunks(text, max_tokens):
    """
    Divide um texto longo em pedaços menores, respeitando o limite aproximado de max_tokens.
    Assume 1 token ≈ 4 caracteres.
    Retorna uma lista de strings (chunks).
    """
    approx_chunk_size = max_tokens * 4  # Aproximação: 1 token ≈ 4 caracteres
    chunks = []
    start = 0
    while start < len(text):
        end = start + approx_chunk_size
        # Garante que não corta no meio de uma palavra
        if end < len(text):
            split_at = text.rfind('\n', start, end)
            if split_at == -1:
                split_at = text.rfind(' ', start, end)
            if split_at == -1 or split_at <= start:
                split_at = end
        else:
            split_at = len(text)
        chunks.append(text[start:split_at].strip())
        start = split_at
    return [c for c in chunks if c]

def is_count_query(text: str) -> bool:
    text_lower = text.lower()
    return any(p in text_lower for p in [
        "quantos documentos", "quantos arquivos", "quantos pdf", "quantos estão indexados",
        "quantos documentos possui", "quantos arquivos possui", "quantos pdfs possui"
    ])

def is_list_query(text: str) -> str | None:
    text_lower = text.lower()
    if "quais pdf" in text_lower or "lista de pdf" in text_lower:
        return "pdf"
    if "quais documentos" in text_lower or "quais arquivos" in text_lower or "lista de documentos" in text_lower:
        return "all"
    return None

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question, agent: Optional[str] = Query(None, description="Nome do agente (domínio)", alias="agent")):
    try:
        # Seleciona o agente
        ag = agent_manager.get_agent(agent)
        # --- NOVO: respostas diretas para perguntas sobre o acervo ---
        if is_count_query(question.text):
            docs = ag.vector_store.index_agent.documents
            pdfs = [doc for doc in docs if doc["url"].lower().endswith(".pdf")]
            if "pdf" in question.text.lower():
                return Answer(answer=f"Atualmente, há {len(pdfs)} PDFs indexados.", sources=[], scores=[])
            return Answer(answer=f"Atualmente, há {len(docs)} documentos indexados.", sources=[], scores=[])
        list_type = is_list_query(question.text)
        if list_type == "pdf":
            docs = [doc for doc in ag.vector_store.index_agent.documents if doc["url"].lower().endswith(".pdf")]
            lista = "\n".join(f"- {doc['url']}" for doc in docs)
            return Answer(answer=f"Os seguintes PDFs estão indexados:\n{lista}", sources=[doc["url"] for doc in docs], scores=[])
        elif list_type == "all":
            docs = ag.vector_store.index_agent.documents
            lista = "\n".join(f"- {doc['url']}" for doc in docs)
            return Answer(answer=f"Os seguintes documentos estão indexados:\n{lista}", sources=[doc["url"] for doc in docs], scores=[])
        index_query_type = is_index_query(question.text)
        if index_query_type:
            docs = ag.vector_store.index_agent.documents
            if index_query_type == "pdf":
                pdfs = [doc for doc in docs if doc["url"].lower().endswith(".pdf")]
                if not pdfs:
                    return Answer(answer="Nenhum PDF está indexado no momento.", sources=[], scores=[])
                lista = "\n".join(f"- {doc['url']}" for doc in pdfs)
                return Answer(
                    answer=f"Os seguintes PDFs estão indexados:\n{lista}",
                    sources=[doc["url"] for doc in pdfs],
                    scores=[]
                )
            else:
                if not docs:
                    return Answer(answer="Nenhum documento está indexado no momento.", sources=[], scores=[])
                lista = "\n".join(f"- {doc['url']}" for doc in docs)
                return Answer(
                    answer=f"Os seguintes documentos estão indexados:\n{lista}",
                    sources=[doc["url"] for doc in docs],
                    scores=[]
                )
        regiao, regionalismo = detectar_regionalismo(question.text)
        query_embedding = ag.get_embedding(question.text)
        keyword = extract_keyword_from_question(question.text)
        relevant_docs = ag.vector_store.search(query_embedding, k=8)
        if keyword:
            filtered_docs = [doc for doc in relevant_docs if keyword in doc["url"].lower()]
            if filtered_docs:
                relevant_docs = filtered_docs
        important_keywords = ['edital', 'relatorio', 'calendario', 'resultado', 'pdf']
        relevant_docs = sorted(
            relevant_docs,
            key=lambda doc: (
                not doc["url"].lower().endswith('.pdf'),
                -sum(kw in doc["url"].lower() or kw in doc["content"].lower() for kw in important_keywords),
                -doc.get('score', 0)
            )
        )
        scores = [doc.get('score', 0.0) for doc in relevant_docs]
        context = "\n\n".join([
            f"Fonte: {doc['url']}\n{doc['content']}" for doc in relevant_docs
        ])
        max_tokens = 1500
        chunks = split_into_chunks(context, max_tokens)
        if len(chunks) > 2:
            top_docs = sorted(zip(relevant_docs, chunks), key=lambda x: x[0].get('score', 0), reverse=True)[:2]
            chunks = [c for d, c in top_docs]
        fallback_msg = "Não foram encontradas informações específicas sobre este documento."
        all_answers = []
        for chunk in chunks:
            prompt = f"""Com base no contexto fornecido sobre a UFPB, responda a pergunta em português.\nSe você não puder responder com base no contexto, responda apenas \"{fallback_msg}\"\n\nContexto:\n{chunk}\n\nPergunta: {question.text}\n\nResposta:"""
            if regiao:
                prompt += f"\n\nAtenção: O usuário utilizou o regionalismo '{regionalismo}' típico da região '{regiao}'. Tente responder utilizando um tom ou expressão similar, se possível."
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.1,
                max_tokens=500
            )
            answer = response.choices[0].message.content.strip()
            all_answers.append(answer)
        fallback_msg = "Não foram encontradas informações específicas sobre este documento."
        explicit_keywords = [kw for kw in keyword.split('-') if kw]
        if explicit_keywords:
            filtered_docs = [
                doc for doc in relevant_docs
                if any(kw in doc["url"].lower() or kw in doc["content"].lower() for kw in explicit_keywords)
            ]
            if filtered_docs:
                relevant_docs = filtered_docs
        important_keywords = ['edital', 'relatorio', 'calendario', 'resultado', 'pdf']
        relevant_docs = sorted(
            relevant_docs,
            key=lambda doc: (
                not doc["url"].lower().endswith('.pdf'),
                -sum(kw in doc["url"].lower() or kw in doc["content"].lower() for kw in important_keywords),
                -doc.get('score', 0)
            )
        )
        sources = list(set(doc["url"].split('#')[0] for doc in relevant_docs))
        good_answers = [a for a in all_answers if a and fallback_msg.lower() not in a.lower()]
        if good_answers:
            final_answer = "\n\n".join(good_answers)
        else:
            final_answer = fallback_msg
        if relevant_docs:
            fontes = "\n".join([f"Fonte: {doc['url']}" for doc in relevant_docs])
            final_answer = f"{final_answer}\n\n{fontes}"
        return Answer(
            answer=final_answer,
            sources=sources,
            scores=[doc.get('score', 0.0) for doc in relevant_docs]
        )
    except Exception as e:
        print("Erro no endpoint /ask:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documentos")
async def list_documents(agent: Optional[str] = Query(None, description="Nome do agente (domínio)", alias="agent")):
    """Lista todos os documentos indexados (URL e preview do conteúdo) para o agente selecionado."""
    ag = agent_manager.get_agent(agent)
    docs = ag.vector_store.index_agent.documents
    return [
        {"url": doc["url"], "preview": doc["content"][:200]} for doc in docs
    ]

# @app.get("/cardapio", response_model=CardapioResponse)
# async def get_cardapio(
#     data: str = Query(None, description="Data no formato YYYY-MM-DD, default hoje"),
#     refeicao: str = Query(None, description="almoço ou jantar, default jantar")
# ):
#     cardapio = consultar_cardapio(data, refeicao)
#     if not cardapio:
#         raise HTTPException(status_code=404, detail="Cardápio não encontrado para os parâmetros informados.")
#     return cardapio

@app.get("/health")
async def health_check():
    return {"status": "healthy"}