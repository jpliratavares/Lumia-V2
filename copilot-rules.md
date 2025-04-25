# Copilot Rules

- Sempre responda em português.
- Priorize exemplos de código claros, comentados e funcionais.
- Use boas práticas de Python, FastAPI e Streamlit.
- Não sugira bibliotecas pagas ou proprietárias.
- Evite respostas ofensivas, inadequadas ou fora do contexto de software.
- Quando possível, explique resumidamente as soluções sugeridas.
- Prefira respostas sucintas e diretas.
- Se houver múltiplas abordagens, sugira a mais simples e robusta.

# Proteção e conduta de código

- Não modifique arquivos fundamentais sem autorização explícita. Exemplos:
  - `main.py`
  - `orchestrator.py`
  - `settings.py`
  - `.env`
  - `database.py`
  - `requirements.txt`
  - Qualquer coisa dentro de `migrations/` ou `core/`

- Nunca remova ou reescreva funções já existentes, a menos que a instrução seja clara para isso.

- Nunca faça refatorações automáticas sem que sejam pedidas diretamente.

- Comente sempre que a sugestão envolver lógica condicional, acesso a APIs, banco de dados ou threads assíncronas.

- Mantenha a nomenclatura coesa com o restante do projeto. Siga padrões já adotados.

- Quando sugerir código novo, não altere o comportamento atual do sistema (preservar compatibilidade).

- Ao lidar com banco de dados, respeite a estrutura existente e sugira alterações somente com validação de impacto.

- Não sugira alterações em testes automatizados sem revisão manual.

- Ao sugerir endpoints ou mudanças de rota, verifique se respeitam a arquitetura atual (como `/api/v1/...`).

# Estilo e clareza

- Sugira sempre estruturas modulares (funções e classes bem separadas).
- Comente de forma curta e útil: o que o bloco faz e por quê.
- Evite comentários redundantes como `# função que retorna um valor`.

# Quando em dúvida

- Pergunte antes de sugerir algo que possa quebrar ou mudar lógica.
- Sempre trate o projeto como um sistema real e em produção.

