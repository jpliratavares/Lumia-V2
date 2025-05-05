# Projeto Lumia V2 - Estrutura de Pastas

- `copilot-rules.md`: Regras e diretrizes para uso do Copilot.
- `requirements.txt`: Dependências do projeto Python.
- `ui.py`: Interface de usuário do projeto.

## assets/
- `logo.png`: Logotipo do projeto.

## data/
- `cardapios.db`: Banco de dados de cardápios.
- `documents.json`: Documentos em formato JSON.
- `embeddings.npy`: Arquivo de embeddings em NumPy.

## src/
- `cardapio_manager.py`: Gerenciamento de cardápios.
    - Principais funções:
        - `consultar_cardapio(data, refeicao, db_path)`: Consulta cardápio por data e refeição.
        - `verificar_ou_atualizar_cardapio_automaticamente()`: Atualiza automaticamente o cardápio do dia, faz download do story do Instagram, executa OCR e salva no banco.
- `main.py`: Arquivo principal de execução.
    - Principais funções:
        - `main()`: Inicializa o sistema, verifica ambiente virtual, carrega dados e inicia o servidor API.
        - `collect_and_index_data()`: Carrega e indexa dados para busca semântica.
        - `check_venv()`: Verifica se está em ambiente virtual.

### src/api/
- `qa_endpoint.py`: Endpoint de perguntas e respostas.
    - Responsável por receber perguntas via API e retornar respostas baseadas em embeddings e busca semântica.

### src/database/
- `vector_store.py`: Gerenciamento de vetores para busca semântica.
    - Principais funções:
        - `load(path)`: Carrega documentos e embeddings.
        - Métodos para salvar e buscar vetores.

### src/scrapers/
- `ufpb_full_scraper.py`: Scraper completo da UFPB.
    - Classe `UFPBFullScraper`:
        - `run()`: Faz crawling em páginas e PDFs, extrai texto e salva embeddings.
        - `save_doc_and_embedding(url, text)`: Salva documento e embedding correspondente.
- `ufpb_scraper.py`: Scraper simples da UFPB.
    - Classe `UFPBScraper`:
        - `scrape_all(max_pages, save_callback)`: Coleta recursiva de páginas, extrai texto e executa callback de salvamento.
- `pdf/`: Pasta para scrapers de PDFs.

---

Arquivos `__pycache__/` são gerados automaticamente pelo Python e armazenam bytecode compilado.