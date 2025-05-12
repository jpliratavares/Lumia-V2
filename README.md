## Metadados de Documentos

O projeto LumIA permite gerar e consumir metadados dos documentos indexados para facilitar buscas, exibições e integrações.

### Como gerar o arquivo de metadados

1. Certifique-se de que o arquivo `scraped_data/documents.json` existe e está atualizado.
2. Execute o script manualmente pelo terminal:
   ```sh
   python generate_metadata.py
   ```
   Isso irá criar (ou atualizar) o arquivo `scraped_data/metadata.json` com os metadados de todos os documentos.

### Como a main consome os metadados

- O arquivo `src/main.py` possui a função utilitária `load_metadata()`, que carrega o `metadata.json` se ele existir.
- O sistema **nunca gera metadados automaticamente**: a main apenas lê o arquivo já existente.
- Se o arquivo não existir, a main avisa no terminal e retorna uma lista vazia.

### Quando atualizar o metadata.json?

- Sempre que novos documentos forem extraídos ou o `documents.json` for alterado.
- Basta rodar novamente o script `generate_metadata.py` para atualizar os metadados.

### Exemplo de uso

```python
from src.main import load_metadata
metadados = load_metadata()
print(f"Total de metadados carregados: {len(metadados)}")
```

Assim, o controle sobre os metadados fica manual, seguro e transparente para o usuário e para o backend.
