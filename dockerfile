# Imagem base compatível com PyTorch e Sentence-Transformers
FROM python:3.10-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Instala dependências básicas do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Atualiza o pip e instala os pacotes Python necessários
RUN pip install --upgrade pip \
    && pip install torch==1.13.1 \
    && pip install sentence-transformers

# Copia o código-fonte do projeto para dentro do container
COPY . .

# Define o ponto de entrada para o seu backend
CMD ["python", "src/main.py"]
