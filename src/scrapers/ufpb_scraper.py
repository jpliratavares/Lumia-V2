import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from urllib.parse import urljoin
import time
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

DATA_DIR = "data"
DOCUMENTS_PATH = os.path.join(DATA_DIR, "documents.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

IGNORE_EXTENSIONS = [
    ".pdf", ".jpg", ".jpeg", ".png", ".css", ".js", ".doc", ".docx", ".xls", ".xlsx",
    ".ppt", ".pptx", ".zip", ".rar", ".mp3", ".mp4", ".avi", ".mov", ".svg", ".xml", ".ico"
]

def has_ignored_extension(url):
    url_lower = url.lower()
    return any(url_lower.endswith(ext) for ext in IGNORE_EXTENSIONS)

class UFPBScraper:
    def __init__(self):
        self.base_url = "https://www.ufpb.br/inova"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def _get_page_content(self, url: str, max_retries: int = 3) -> str:
        """Get the content of a page with retries"""
        print(f"\nTentando acessar: {url}")
        for attempt in range(max_retries):
            try:
                print(f"  Tentativa {attempt + 1}/{max_retries}...")
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 404:
                    print(f"  ❌ Página não encontrada (404)")
                    return ""
                response.raise_for_status()
                print(f"  ✅ Página acessada com sucesso")
                return response.text
            except requests.RequestException as e:
                print(f"  ❌ Erro na tentativa {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    print("  ⏳ Aguardando 1 segundo antes da próxima tentativa...")
                    time.sleep(1)
                continue
            except Exception as e:
                print(f"  ❌ Erro inesperado: {str(e)}")
                return ""
        print("  ❌ Todas as tentativas falharam")
        return ""

    def _extract_direct_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract direct links from the page (not following links within links)"""
        print("\nBuscando links na página...")
        links = []
        try:
            for a in soup.find_all('a', href=True):
                href = a['href']
                full_url = urljoin(base_url, href)
                
                # Only include links from the same domain and avoid RSS feeds
                if full_url.startswith(self.base_url) and 'rss' not in full_url:
                    links.append(full_url)
            
            print(f"  ✅ Encontrados {len(links)} links únicos")
            for link in links:
                print(f"    - {link}")
        except Exception as e:
            print(f"  ❌ Erro ao extrair links: {str(e)}")
        
        return list(set(links))  # Remove duplicates

    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract meaningful text content from the page"""
        try:
            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                script.decompose()
            
            # Get text and clean it
            text = ' '.join(soup.stripped_strings)
            content = text.strip()
            if content:
                print("  ✅ Conteúdo extraído com sucesso")
                print(f"    - Tamanho do conteúdo: {len(content)} caracteres")
            else:
                print("  ⚠️ Nenhum conteúdo significativo encontrado")
            return content
        except Exception as e:
            print(f"  ❌ Erro ao extrair conteúdo: {str(e)}")
            return ""

    def scrape_all(self, max_pages: int = 1000, save_callback=None) -> List[Dict[str, str]]:
        """Crawl recursivamente todas as páginas do domínio base da UFPB, salvando e limpando cache após cada página."""
        print("\n🔍 Iniciando coleta recursiva de dados da UFPB...")
        processed_urls = set()
        to_visit = [self.base_url]
        page_count = 0
        
        while to_visit and page_count < max_pages:
            url = to_visit.pop(0)
            if url in processed_urls:
                continue
            # IGNORA arquivos binários
            if has_ignored_extension(url):
                print(f"  ⚠️ Ignorando arquivo binário: {url}")
                processed_urls.add(url)
                continue
            print(f"\n[{page_count+1}] Visitando: {url}")
            content = self._get_page_content(url)
            if not content:
                processed_urls.add(url)
                continue
            try:
                soup = BeautifulSoup(content, 'html.parser')
            except Exception as e:
                print(f"  ❌ Erro ao processar conteúdo da página (provavelmente não HTML): {e}")
                processed_urls.add(url)
                continue
            text = self._extract_text_content(soup)
            if text:
                doc = {'url': url, 'content': text}
                if save_callback:
                    save_callback(doc)
            processed_urls.add(url)
            page_count += 1
            links = self._extract_direct_links(soup, url)
            for link in links:
                if link not in processed_urls and link not in to_visit:
                    to_visit.append(link)
            if to_visit:
                print("  ⏳ Aguardando 0.5 segundos antes da próxima página...")
                time.sleep(0.5)
            # Limpa variáveis pesadas para liberar RAM
            del soup, content, text
        print(f"\n✨ Coleta recursiva finalizada!")
        print(f"📊 Estatísticas:")
        print(f"   - Total de páginas processadas: {len(processed_urls)}")
        return []

if __name__ == "__main__":
    scraper = UFPBScraper()
    scraper.scrape_all()