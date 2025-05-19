import os
import json
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import tempfile
import PyPDF2
import faiss

DATA_DIR = "data"
DOCS_PATH = os.path.join(DATA_DIR, "documents.json")
FAISS_PATH = os.path.join(DATA_DIR, "faiss.index")
VISITED_PATH = os.path.join(DATA_DIR, "visited.json")
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

class SimpleFullScraper:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip('/')
        self.model = SentenceTransformer(MODEL_NAME)
        self.documents = []
        self.embeddings = []
        self.visited = set()
        os.makedirs(DATA_DIR, exist_ok=True)
        self.index = None
        self._load_existing()
        # Sincroniza visited.json com todas as URLs já presentes em documents.json
        if self.documents:
            doc_urls = {doc['url'] for doc in self.documents if 'url' in doc}
            if not self.visited.issuperset(doc_urls):
                self.visited.update(doc_urls)
                with open(VISITED_PATH, 'w', encoding='utf-8') as f:
                    json.dump(list(self.visited), f, ensure_ascii=False, indent=2)

    def _load_existing(self):
        if os.path.exists(DOCS_PATH):
            with open(DOCS_PATH, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
        if os.path.exists(FAISS_PATH):
            self.index = faiss.read_index(FAISS_PATH)
        else:
            self.index = faiss.IndexFlatL2(384)  # 384 dims for MiniLM
        if os.path.exists(VISITED_PATH):
            with open(VISITED_PATH, 'r', encoding='utf-8') as f:
                try:
                    self.visited = set(json.load(f))
                except Exception:
                    self.visited = set()

    def _save(self):
        with open(DOCS_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        faiss.write_index(self.index, FAISS_PATH)
        with open(VISITED_PATH, 'w', encoding='utf-8') as f:
            json.dump(list(self.visited), f, ensure_ascii=False, indent=2)

    def _extract_pdf_text(self, url):
        try:
            with requests.get(url, stream=True, timeout=15) as r:
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
            print(f"[PDF ERROR] {url}: {e}")
            return None

    def _extract_html_text(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        # Não discrimina: pega todo o texto visível
        return soup.get_text(separator=' ', strip=True)

    def _is_valid_url(self, url):
        parsed = urlparse(url)
        base = urlparse(self.base_url)
        return parsed.scheme in ['http', 'https'] and (parsed.netloc.endswith(base.netloc) or parsed.netloc == base.netloc)

    def run(self, max_pages=10000):
        to_visit = [self.base_url]
        count = 0
        while to_visit and count < max_pages:
            url = to_visit.pop(0)
            if url in self.visited:
                continue
            print(f"[SCRAPER] Visitando: {url}")
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                content_type = resp.headers.get('Content-Type', '')
                if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                    text = self._extract_pdf_text(url)
                elif 'text/html' in content_type:
                    text = self._extract_html_text(resp.text)
                else:
                    text = None
                if text and len(text.strip()) > 0:
                    doc = {'url': url, 'content': text}
                    self.documents.append(doc)
                    emb = self.model.encode([text], convert_to_numpy=True)
                    self.index.add(emb)
                    self._save()
                    print(f"[SCRAPER] Documento salvo e embedding adicionado: {url}")
                if 'text/html' in content_type:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        next_url = urljoin(url, link['href'])
                        if self._is_valid_url(next_url) and next_url not in self.visited and next_url not in to_visit:
                            to_visit.append(next_url)
                self.visited.add(url)
                count += 1
            except Exception as e:
                print(f"[SCRAPER ERROR] {url}: {e}")
        self._save()
        print(f"[SCRAPER] Finalizado. Total de documentos: {len(self.documents)}")

if __name__ == "__main__":
    base_url = "https://www.ufpb.br/"
    scraper = SimpleFullScraper(base_url)
    scraper.run()
