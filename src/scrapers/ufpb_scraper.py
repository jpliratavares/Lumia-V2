import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from urllib.parse import urljoin
import time

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

    def scrape_all(self) -> List[Dict[str, str]]:
        """Scrape the main page and its direct links"""
        print("\n🔍 Iniciando coleta de dados da UFPB...")
        documents = []
        processed_urls = set()  # Para evitar URLs duplicadas
        
        print("\n📄 Coletando página principal...")
        main_content = self._get_page_content(self.base_url)
        if main_content:
            main_soup = BeautifulSoup(main_content, 'html.parser')
            
            # Add main page content
            print("\nExtraindo conteúdo da página principal...")
            main_text = self._extract_text_content(main_soup)
            if main_text:
                documents.append({
                    'url': self.base_url,
                    'content': main_text
                })
                processed_urls.add(self.base_url)
                print("  ✅ Página principal processada")
            
            # Get direct links
            direct_links = self._extract_direct_links(main_soup, self.base_url)
            
            if direct_links:
                print(f"\n📑 Processando {len(direct_links)} páginas encontradas...")
                # Scrape each direct link
                for i, link in enumerate(direct_links, 1):
                    print(f"\n[{i}/{len(direct_links)}] Processando página...")
                    if link not in processed_urls:
                        content = self._get_page_content(link)
                        if content:
                            soup = BeautifulSoup(content, 'html.parser')
                            text = self._extract_text_content(soup)
                            if text:
                                documents.append({
                                    'url': link,
                                    'content': text
                                })
                                print("  ✅ Página processada com sucesso")
                        processed_urls.add(link)
                        if i < len(direct_links):
                            print("  ⏳ Aguardando 0.5 segundos antes da próxima página...")
                            time.sleep(0.5)
            else:
                print("\n⚠️ Nenhum link adicional encontrado")
        
        print(f"\n✨ Coleta finalizada!")
        print(f"📊 Estatísticas:")
        print(f"   - Total de páginas processadas: {len(processed_urls)}")
        print(f"   - Total de documentos coletados: {len(documents)}")
        return documents