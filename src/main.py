import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scrapers.ufpb_scraper import UFPBScraper
from database.vector_store import VectorStore
from sentence_transformers import SentenceTransformer
from cardapio_manager import verificar_ou_atualizar_cardapio_automaticamente
verificar_ou_atualizar_cardapio_automaticamente()
import uvicorn
from api.qa_endpoint import app

def check_venv():
    """Verifica se o código está rodando em um ambiente virtual"""
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        print("\n⚠️  AVISO: O código não está rodando em um ambiente virtual!")
        print("Para ativar o ambiente virtual, use um dos comandos:")
        print("  Windows PowerShell: .\\venv\\Scripts\\Activate.ps1")
        print("  Windows CMD: .\\venv\\Scripts\\activate.bat")
        sys.exit(1)

def collect_and_index_data():
    vector_store = VectorStore()
    # Tenta carregar dados existentes primeiro
    try:
        print("Tentando carregar dados existentes...")
        vector_store.load("data")
        print(f"Dados carregados com sucesso! {len(vector_store.documents)} documentos encontrados.")
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

if __name__ == "__main__":
    main()