# Mindlog - LumIA V2

## [2025-05-05]

### Problemas e desafios encontrados:
- Erro 413 (payload too large) ao enviar contexto grande para o Groq/LLM.
- Respostas do modelo misturando fallback negativo com respostas corretas.
- Timeout no endpoint /ask devido a muitos chunks ou contexto extenso.
- Dificuldade do usuário em controlar mudanças e histórico do projeto devido ao tamanho do código.
- Falta de mecanismo para listar e filtrar documentos indexados de forma inteligente via /ask.
- Sistema de scraping de stories do Instagram bloqueando execução por checkpoint/login.

### Linha de raciocínio e soluções:
- Refatoração do vector_store.py para arquitetura de agentes, facilitando modularidade e manutenção.
- Implementação de split_into_chunks para dividir contexto e evitar erro 413.
- Limitação do número de chunks processados para evitar timeout.
- Filtro opcional por palavra-chave extraída da pergunta para restringir busca por URL.
- Melhoria do fallback: só retorna mensagem padrão se todas as respostas forem negativas.
- Criação de endpoint /documentos para depuração e transparência do acervo.
- Desativação temporária do scraping de stories do Instagram para evitar bloqueios.
- Criação deste mindlog.md para registrar problemas, raciocínio e evitar repetição de ações.

### Próximos passos sugeridos:
- Sempre consultar changelog.md e mindlog.md antes de qualquer alteração.
- Automatizar testes para garantir que respostas do modelo nunca misturem fallback com respostas boas.
- Avaliar performance do filtro por palavra-chave e ajustar regex/normalização se necessário.
- Considerar paginação ou sumarização para respostas com muitos documentos.
