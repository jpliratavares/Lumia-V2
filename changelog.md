# Changelog - LumIA V2

## [2025-05-05]
- Refatoração do vector_store.py para arquitetura de agentes (IndexAgent, SearchAgent, Orchestrator).
- Adição de .gitignore para proteger .env, arquivos de dados, cache e configs do VSCode.
- Criação de endpoint /documentos para listar documentos indexados.
- Ajuste do carregamento de dados para priorizar scraped_data/ se existir.
- Correção do main.py para acessar vector_store.index_agent.documents.
- Implementação de split_into_chunks para dividir contexto em chunks menores e evitar erro 413 do Groq.
- Limite de 2 chunks processados por requisição para evitar timeout.
- Filtro opcional por palavra-chave extraída da pergunta para restringir busca por URL.
- Ajuste do fallback: só retorna mensagem padrão se todas as respostas forem negativas.
- Melhoria do endpoint /ask para responder perguntas sobre o acervo (quais arquivos/PDFs estão indexados).
- Desativação temporária do sistema de obtenção de stories do Instagram.
- Adição de fallback seguro para respostas do modelo.
- Melhoria na lógica de resposta para não misturar fallback com respostas boas.
