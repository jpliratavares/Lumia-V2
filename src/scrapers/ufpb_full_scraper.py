import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

class UFPBFullScraper:
    def __init__(self, base_url, data_dir):
        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(base_url).netloc
        self.visited = set()
        self.to_visit = set([self.base_url])
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.documents_path = os.path.join(self.data_dir, "documents.json")
        self.embeddings_path = os.path.join(self.data_dir, "embeddings.npy")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def is_valid_url(self, url):
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and parsed.netloc == self.domain

    def extract_text(self, soup):
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript', 'svg', 'form', 'aside']):
            tag.decompose()
        texts = soup.stripped_strings
        return [t for t in texts if len(t.split()) >= 5]

    def save_doc_and_embedding(self, url, text):
        doc = {"url": url, "content": text}
        # Salva documento
        if os.path.exists(self.documents_path):
            with open(self.documents_path, "r", encoding="utf-8") as f:
                docs = json.load(f)
        else:
            docs = []
        docs.append(doc)
        with open(self.documents_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False)
        # Gera embedding e salva incrementalmente
        emb = self.model.encode([text], convert_to_tensor=False)
        if os.path.exists(self.embeddings_path):
            embs = np.load(self.embeddings_path)
            embs = np.vstack([embs, emb])
        else:
            embs = emb
        np.save(self.embeddings_path, embs)
        del emb

    def run(self, max_pages=None, delay=0.5, max_retries=3):
        count = 0
        failed_urls = set()
        while True:
            if not self.to_visit:
                # Se não há mais URLs para visitar, mas há falhas, tente novamente as falhas
                if failed_urls:
                    print(f"Re-tentando {len(failed_urls)} URLs que falharam anteriormente...")
                    self.to_visit = failed_urls.copy()
                    failed_urls.clear()
                    time.sleep(1)  # Espera antes de tentar de novo
                else:
                    break  # Não há mais nada para tentar
            url = self.to_visit.pop()
            if url in self.visited:
                continue
            retries = 0
            success = False
            while retries < max_retries:
                try:
                    resp = requests.get(url, timeout=15)
                    if resp.status_code != 200 or 'text/html' not in resp.headers.get('Content-Type', ''):
                        break
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    texts = self.extract_text(soup)
                    for text in texts:
                        self.save_doc_and_embedding(url, text)
                    for link in soup.find_all('a', href=True):
                        next_url = urljoin(url, link['href'])
                        if self.is_valid_url(next_url) and next_url not in self.visited:
                            self.to_visit.add(next_url)
                    success = True
                    break  # Sucesso, não precisa de mais tentativas
                except Exception as e:
                    retries += 1
                    print(f"Erro ao acessar {url} (tentativa {retries}/{max_retries}): {e}")
                    time.sleep(2)
            if not success:
                failed_urls.add(url)
            self.visited.add(url)
            count += 1
            if max_pages and count >= max_pages:
                break
            time.sleep(delay)

if __name__ == "__main__":
    scraper = UFPBFullScraper(
        base_url="https://ufpb.br/",
        data_dir="data"
    )
    scraper.run()
