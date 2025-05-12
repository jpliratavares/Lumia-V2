import json

# Caminho para o arquivo
file_path = 'data/documents.json'  # agora busca sempre em /data

# Carregar os dados
with open(file_path, 'r', encoding='utf-8') as f:
    documents = json.load(f)

# Contar PDFs
pdf_count = sum(1 for doc in documents if doc['url'].lower().endswith('.pdf'))
total_count = len(documents)

print(f"Total de documentos: {total_count}")
print(f"Total de PDFs: {pdf_count}")
