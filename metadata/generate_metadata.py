import json
import os
from datetime import datetime

# Caminho do arquivo de entrada e saída
DOCS_PATH = os.path.join('data', 'documents.json')
META_PATH = os.path.join('data', 'metadata.json')

def clean_text(text):
    """Remove quebras de linha e espaços duplicados."""
    return ' '.join(text.replace('\n', ' ').replace('\r', ' ').split())

def generate_metadata():
    if not os.path.exists(DOCS_PATH):
        print(f"Arquivo {DOCS_PATH} não encontrado.")
        return
    with open(DOCS_PATH, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    metadata = []
    now = datetime.now().isoformat()
    for idx, doc in enumerate(docs, 1):
        url = doc.get('url', None)
        file_name = url.split('/')[-1] if url else f'doc_{idx}'
        content = doc.get('content', '')
        summary = clean_text(content)[:500]
        meta = {
            'id': idx,
            'file_name': file_name,
            'url': url,
            'summary': summary,
            'date_added': now
        }
        metadata.append(meta)
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"{len(metadata)} metadados gerados e salvos em {META_PATH}.")

if __name__ == "__main__":
    generate_metadata()
