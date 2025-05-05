import os
import json
import requests
from datetime import datetime
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import tempfile
import PyPDF2

class UFPBScraper:
    def __init__(self, base_url="https://www.ufpb.br/", log_path="scraper_log.txt", checkpoint_path="checkpoint.json", website_log_path="website_logs.json"):
        self.base_url = base_url
        self.log_path = log_path
        self.checkpoint_path = checkpoint_path
        self.website_log_path = website_log_path
        self.visited_urls = self.load_checkpoint()
        self.website_logs = self.load_website_logs()
        os.makedirs('scraped_data', exist_ok=True)

    def load_checkpoint(self):
        """Load the checkpoint file to get the list of visited URLs."""
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'r') as f:
                return json.load(f)
        return []

    def load_website_logs(self):
        if os.path.exists(self.website_log_path):
            with open(self.website_log_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def save_checkpoint(self):
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.visited_urls, f, indent=2)

    def save_website_logs(self):
        with open(self.website_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.website_logs, f, ensure_ascii=False, indent=2)

    def log_status(self, url, message, error=None):
        """Log the status of the scraping process with a timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, 'a') as f:
            if error:
                f.write(f"{timestamp} - ERROR: {url} - {message}: {error}\n")
            else:
                f.write(f"{timestamp} - INFO: {url} - {message}\n")

    def log_website(self, url, status, message=None):
        log_entry = {"url": url, "status": status, "message": message, "timestamp": datetime.now().isoformat()}
        self.website_logs.append(log_entry)
        self.save_website_logs()
        print(f"[{status.upper()}] {url} - {message if message else ''}")

    def run(self, max_pages=1000, delay=0.5):
        """Percorre recursivamente o domínio base, subdomínios e baixa PDFs, extraindo texto e embeddings."""
        to_visit = [self.base_url]
        processed = set(self.visited_urls)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings_path = os.path.join('scraped_data', 'embeddings.npy')
        docs_path = os.path.join('scraped_data', 'documents.json')
        all_docs = []
        if os.path.exists(docs_path):
            with open(docs_path, 'r', encoding='utf-8') as f:
                all_docs = json.load(f)
        all_embs = None
        if os.path.exists(embeddings_path):
            all_embs = np.load(embeddings_path)
        while to_visit and len(processed) < max_pages:
            url = to_visit.pop(0)
            if url in processed:
                continue
            try:
                print(f"Visitando: {url}")
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                content_type = resp.headers.get('Content-Type', '')
                if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                    text = self._extract_pdf_text_from_url(url)
                    if text:
                        doc = {'url': url, 'content': text}
                        all_docs.append(doc)
                        emb = model.encode([text], convert_to_tensor=False)
                        if all_embs is not None:
                            all_embs = np.vstack([all_embs, emb])
                        else:
                            all_embs = emb
                        self.log_status(url, 'PDF extraído e embedding gerado')
                        self.log_website(url, 'success', 'PDF extraído e embedding gerado')
                    else:
                        self.log_website(url, 'failed', 'Falha ao extrair texto do PDF')
                elif 'text/html' in content_type:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    text = self._extract_text_content(soup)
                    if text:
                        doc = {'url': url, 'content': text}
                        all_docs.append(doc)
                        emb = model.encode([text], convert_to_tensor=False)
                        if all_embs is not None:
                            all_embs = np.vstack([all_embs, emb])
                        else:
                            all_embs = emb
                        self.log_status(url, 'HTML extraído e embedding gerado')
                        self.log_website(url, 'success', 'HTML extraído e embedding gerado')
                    else:
                        self.log_website(url, 'failed', 'Falha ao extrair texto HTML')
                    # Descobre novos links (incluindo subdomínios e PDFs)
                    for link in soup.find_all('a', href=True):
                        next_url = urljoin(url, link['href'])
                        if self._is_valid_url(next_url) and next_url not in processed and next_url not in to_visit:
                            to_visit.append(next_url)
                else:
                    self.log_website(url, 'failed', f'Content-Type não suportado: {content_type}')
                processed.add(url)
                self.visited_urls.append(url)
                self.save_checkpoint()
            except Exception as e:
                self.log_status(url, 'Erro ao processar', error=str(e))
                self.log_website(url, 'failed', f'Exception: {e}')
        # Salva resultados
        with open(docs_path, 'w', encoding='utf-8') as f:
            json.dump(all_docs, f, ensure_ascii=False, indent=2)
        if all_embs is not None:
            np.save(embeddings_path, all_embs)

    def _extract_pdf_text_from_url(self, url):
        try:
            with requests.get(url, stream=True, timeout=10) as r:
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
            self.log_status(url, 'Erro ao extrair texto do PDF', error=str(e))
            return None

    def _extract_text_content(self, soup):
        # Remove scripts, styles e navegação
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return text

    def _is_valid_url(self, url):
        parsed = urlparse(url)
        base = urlparse(self.base_url)
        # Permite subdomínios e PDFs
        return parsed.scheme in ['http', 'https'] and (parsed.netloc.endswith(base.netloc) or parsed.netloc == base.netloc)

if __name__ == "__main__":
    print("Iniciando o scraper UFPBScraper...")
    scraper = UFPBScraper()
    scraper.run(max_pages=20000, delay=0.5)
    print("Scraping finalizado!")