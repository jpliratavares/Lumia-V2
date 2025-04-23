import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scrapers.ufpb_scraper import UFPBScraper
from database.vector_store import VectorStore
from sentence_transformers import SentenceTransformer
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
        print(f"Nenhum dado existente encontrado ou erro ao carregar: {str(e)}")
        print("Iniciando nova coleta de dados...")
    
    # Initialize components for new data collection
    scraper = UFPBScraper()
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print("Coletando dados do site da UFPB...")
    documents = scraper.scrape_all()
    
    if not documents:
        print("Nenhum documento foi encontrado!")
        return None
    
    print(f"Encontrados {len(documents)} documentos.")
    print("Gerando embeddings...")
    
    # Generate embeddings for all documents
    texts = [doc["content"] for doc in documents]
    embeddings = model.encode(texts, convert_to_tensor=False)
    
    print("Salvando no banco de dados vetorial...")
    vector_store.add_documents(documents, embeddings)
    
    # Save the vector store
    os.makedirs("data", exist_ok=True)
    vector_store.save("data")
    
    print("Indexação concluída com sucesso!")
    return vector_store

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