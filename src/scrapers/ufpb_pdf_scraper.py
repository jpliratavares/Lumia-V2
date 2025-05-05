import os
import json
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import tempfile
import PyPDF2

def find_pdf_links(base_url, max_pages=1000):
    """Percorre recursivamente o site e retorna todos os links diretos para PDFs."""
    to_visit = [base_url]
    visited = set()
    pdf_links = set()
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            content_type = resp.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    next_url = urljoin(url, href)
                    if next_url.lower().endswith('.pdf'):
                        pdf_links.add(next_url)
                    elif next_url not in visited and urlparse(next_url).netloc.endswith(urlparse(base_url).netloc):
                        to_visit.append(next_url)
        except Exception as e:
            print(f"Erro ao processar {url}: {e}")
        visited.add(url)
    return list(pdf_links)

def extract_pdf_text(pdf_url):
    try:
        with requests.get(pdf_url, stream=True, timeout=15) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                for chunk in r.iter_content(1024):
                    tmp_pdf.write(chunk)
                tmp_pdf_path = tmp_pdf.name
        with open(tmp_pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = '\n'.join(page.extract_text() or '' for page in reader.pages)
        os.remove(tmp_pdf_path)
        return text.strip()
    except Exception as e:
        print(f"Erro ao extrair texto do PDF {pdf_url}: {e}")
        return None

def main():
    base_url = "https://www.ufpb.br/"
    output_dir = "scraped_data"
    os.makedirs(output_dir, exist_ok=True)
    docs_path = os.path.join(output_dir, "documents.json")
    embs_path = os.path.join(output_dir, "embeddings.npy")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    all_docs = []
    all_embs = None
    print("Buscando links de PDFs...")
    pdf_links = find_pdf_links(base_url)
    print(f"Encontrados {len(pdf_links)} PDFs.")
    for pdf_url in pdf_links:
        print(f"Processando: {pdf_url}")
        text = extract_pdf_text(pdf_url)
        if text and len(text) > 100:
            doc = {"url": pdf_url, "content": text}
            all_docs.append(doc)
            emb = model.encode([text], convert_to_tensor=False)
            if all_embs is not None:
                all_embs = np.vstack([all_embs, emb])
            else:
                all_embs = emb
    with open(docs_path, 'w', encoding='utf-8') as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)
    if all_embs is not None:
        np.save(embs_path, all_embs)
    print(f"Extração finalizada. PDFs processados: {len(all_docs)}")

if __name__ == "__main__":
    main()
