import pytesseract
from PIL import Image
import re
import sqlite3
from datetime import datetime
import instaloader
import shutil
import os
from dotenv import load_dotenv

load_dotenv()

pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# --- OCR ---
def extrair_texto(img_path: str) -> str:
    """Extrai texto de uma imagem usando pytesseract."""
    img = Image.open(img_path)
    texto = pytesseract.image_to_string(img, lang='por')
    return texto

# --- Parsing ---
def parsear_cardapio(texto: str) -> dict:
    """Processa o texto extraído e retorna um dicionário estruturado do cardápio."""
    # Regex e heurísticas simples para extrair campos
    data_match = re.search(r'(\d{2}/\d{2}/\d{4})', texto)
    data = data_match.group(1) if data_match else datetime.now().strftime('%Y-%m-%d')
    data = datetime.strptime(data, '%d/%m/%Y').strftime('%Y-%m-%d') if '/' in data else data

    refeicao = 'jantar' if re.search(r'jantar', texto, re.IGNORECASE) else 'almoço'
    campus_match = re.search(r'campus\s*([\w\s]+)', texto, re.IGNORECASE)
    campus = campus_match.group(1).strip() if campus_match else 'Campus I'

    def extrair_bloco(label):
        padrao = rf'{label}:(.*?)(?:\n\w+:|$)'
        match = re.search(padrao, texto, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip().replace('\n', ', ') if match else ''

    salada = extrair_bloco('salada')
    prato_principal = extrair_bloco('prato principal')
    acompanhamentos = extrair_bloco('acompanhamentos')
    bebidas = extrair_bloco('bebidas')

    return {
        'data': data,
        'refeicao': refeicao,
        'campus': campus,
        'salada': salada,
        'prato_principal': prato_principal,
        'acompanhamentos': acompanhamentos,
        'bebidas': bebidas
    }

# --- Banco de Dados ---
def inicializar_banco(db_path='data/cardapios.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS cardapios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data TEXT,
        refeicao TEXT,
        campus TEXT,
        salada TEXT,
        prato_principal TEXT,
        acompanhamentos TEXT,
        bebidas TEXT
    )''')
    conn.commit()
    conn.close()

# --- Salvamento ---
def salvar_cardapio_no_banco(dados_cardapio: dict, db_path='data/cardapios.db'):
    inicializar_banco(db_path)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''INSERT INTO cardapios (data, refeicao, campus, salada, prato_principal, acompanhamentos, bebidas)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (
                  dados_cardapio['data'],
                  dados_cardapio['refeicao'],
                  dados_cardapio['campus'],
                  dados_cardapio['salada'],
                  dados_cardapio['prato_principal'],
                  dados_cardapio['acompanhamentos'],
                  dados_cardapio['bebidas']
              ))
    conn.commit()
    conn.close()

# --- Consulta ---
def consultar_cardapio(data=None, refeicao=None, db_path='data/cardapios.db'):
    inicializar_banco(db_path)
    if not data:
        data = datetime.now().strftime('%Y-%m-%d')
    if not refeicao:
        refeicao = 'jantar'
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''SELECT data, refeicao, campus, salada, prato_principal, acompanhamentos, bebidas
                 FROM cardapios WHERE data=? AND refeicao=?''', (data, refeicao))
    row = c.fetchone()
    conn.close()
    if row:
        return dict(zip(['data', 'refeicao', 'campus', 'salada', 'prato_principal', 'acompanhamentos', 'bebidas'], row))
    return None

def verificar_ou_atualizar_cardapio_automaticamente():
    """
    Verifica se já existe cardápio para hoje. Se não, apaga antigos, baixa o story do RU,
    faz OCR, parsing e salva o novo cardápio no banco.
    """
    hoje = datetime.now().strftime('%Y-%m-%d')
    cardapio_hoje = consultar_cardapio(hoje)
    if cardapio_hoje:
        return  # Já existe cardápio atualizado

    # Apaga todos os registros antigos
    db_path = 'data/cardapios.db'
    inicializar_banco(db_path)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('DELETE FROM cardapios')
    conn.commit()
    conn.close()

    # Baixa o story mais recente do perfil ru.ufpb
    insta_user = os.getenv('INSTA_USER')
    insta_pass = os.getenv('INSTA_PASS')
    L = instaloader.Instaloader(dirname_pattern='data', download_video_thumbnails=False, save_metadata=False)
    try:
        L.login(insta_user, insta_pass)
        stories = L.get_stories(userids=[L.check_profile_id('ru.ufpb')])
        img_path = None
        for story in stories:
            for item in story.get_items():
                if item.is_video:
                    continue
                # Salva a imagem mais recente
                img_path = L.download_storyitem(item, 'data')
                break
            if img_path:
                break
        # Renomeia/copia para nome fixo
        if img_path:
            final_path = 'data/cardapio_mais_recente.jpg'
            shutil.copy(img_path, final_path)
        else:
            print('Nenhuma imagem de story encontrada no perfil ru.ufpb.')
            return
    except Exception as e:
        print(f'Erro ao baixar story do Instagram: {e}')
        return

    # OCR, parsing e salvamento
    texto = extrair_texto('data/cardapio_mais_recente.jpg')
    dados = parsear_cardapio(texto)
    salvar_cardapio_no_banco(dados)
