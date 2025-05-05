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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.vector_store import VectorStore
from cardapio_manager import consultar_cardapio

load_dotenv()

app = FastAPI()
vector_store = VectorStore()
try:
    # Carrega dados de scraped_data se existir, senão de data
    if os.path.exists(os.path.join("scraped_data", "documents.json")) and os.path.exists(os.path.join("scraped_data", "embeddings.npy")):
        vector_store.load("scraped_data")
        print("[QA] Dados carregados de scraped_data/")
    else:
        vector_store.load("data")
        print("[QA] Dados carregados de data/")
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

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    try:
        # Detecta regionalismo na pergunta
        regiao, regionalismo = detectar_regionalismo(question.text)
        
        # Get question embedding
        query_embedding = get_embedding(question.text)
        
        # Search for relevant documents with lower threshold and more results
        relevant_docs = vector_store.search(query_embedding, k=5)  # Aumentado de 3 para 5
        
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

        # Se detectar regionalismo, sugere adaptar a resposta
        if regiao:
            prompt += f"\n\nAtenção: O usuário utilizou o regionalismo '{regionalismo}' típico da região '{regiao}'. Tente responder utilizando um tom ou expressão similar, se possível."

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

@app.get("/cardapio", response_model=CardapioResponse)
async def get_cardapio(
    data: str = Query(None, description="Data no formato YYYY-MM-DD, default hoje"),
    refeicao: str = Query(None, description="almoço ou jantar, default jantar")
):
    cardapio = consultar_cardapio(data, refeicao)
    if not cardapio:
        raise HTTPException(status_code=404, detail="Cardápio não encontrado para os parâmetros informados.")
    return cardapio

@app.get("/documentos")
async def list_documents():
    """Lista todos os documentos indexados (URL e preview do conteúdo)."""
    docs = vector_store.documents
    return [
        {"url": doc["url"], "preview": doc["content"][:200]} for doc in docs
    ]

@app.get("/health")
async def health_check():
    return {"status": "healthy"}