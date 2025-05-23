import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scrapers.ufpb_scraper import UFPBScraper
from fastapi import FastAPI, HTTPException
from database.vector_store import VectorStore
from sentence_transformers import SentenceTransformer
# Removido: verificar_ou_atualizar_cardapio_automaticamente()
import uvicorn
from api.qa_endpoint import router as qa_router
from api.qb_agent import router as qb_router  # ou src.agents.qb_agent dependendo do caminho
from api.ask_router import router as ask_router

app = FastAPI()

app.include_router(qa_router)
app.include_router(qb_router)
app.include_router(ask_router)


def check_venv():
    """Verifica se o código está rodando em um ambiente virtual"""
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        print("\n⚠️  AVISO: O código não está rodando em um ambiente virtual!")
        print("Para ativar o ambiente virtual, use um dos comandos:")
        print("  Windows PowerShell: .\\venv\\Scripts\\Activate.ps1")
        print("  Windows CMD: .\\venv\\Scripts\\activate.bat")
        print("  Linux/Mac: source venv/bin/activate")
        print("Recomenda-se usar o ambiente virtual para evitar conflitos de dependências.")
        print("Continuando a execução do código pro gostoso do JP ser feliz finalmente sem a porcaria do erro do WSL...\n")


def load_metadata():
    """
    Carrega o arquivo metadata.json se existir.
    Retorna uma lista de dicionários com os metadados.
    Se não existir, retorna lista vazia e avisa no terminal.
    """
    meta_path = os.path.join('data', 'metadata.json')
    if not os.path.exists(meta_path):
        print("metadata.json não encontrado, rode generate_metadata.py primeiro.")
        return []
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def collect_and_index_data():
    vector_store = VectorStore()
    # Tenta carregar dados existentes primeiro
    try:
        print("Tentando carregar dados existentes...")
        vector_store.load("data")
        # Compatível com arquitetura de agentes
        num_docs = len(vector_store.index_agent.documents)
        print(f"Dados carregados com sucesso! {num_docs} documentos encontrados.")
        return vector_store
    except Exception as e:
        print(f"Erro ao carregar dados existentes: {str(e)}")
        print("Os dados precisam ser coletados previamente usando o scraper.")
        return None

def main():
    # Verifica ambiente virtual
    check_venv()
    
    # First collect and index the data
    vector_store = collect_and_index_data()
    
    if not vector_store:
        print("Erro durante a indexação dos dados!")
        return
    
    print("\nIniciando servidor API...")
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Adiciona isso no final do main.py
if __name__ == "__main__":
    main()
else:
    # Permite que o uvicorn acesse a variável app diretamente
    check_venv()
    collect_and_index_data()
