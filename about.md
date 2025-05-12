# Sobre o Projeto Lumia

Lumia é uma assistente de IA desenvolvida para facilitar o acesso a informações institucionais, com foco inicial na Universidade Federal da Paraíba (UFPB), mas projetada para ser facilmente adaptável a outras universidades e, futuramente, ao setor empresarial (Lumia For Business).

## Objetivo

O Lumia visa democratizar e agilizar o acesso a informações públicas e documentos institucionais, utilizando busca vetorial, processamento de linguagem natural e um banco de regionalismos para entender perguntas em diferentes formas de escrita e sotaques regionais.

## Componentes Principais

- **Scrapers Inteligentes:** Extraem de forma eficiente todo o conteúdo relevante (PDFs, HTML, editais, relatórios, etc.) do site da instituição-alvo, garantindo cobertura máxima e evitando duplicidade.
- **Indexação Semântica:** Todo o conteúdo extraído é convertido em embeddings vetoriais usando um modelo de linguagem consistente, permitindo buscas semânticas robustas e rápidas.
- **Banco de Regionalismos:** Um dicionário extenso de expressões regionais brasileiras, permitindo que o sistema compreenda e responda perguntas feitas em diferentes dialetos e gírias.
- **API de Perguntas e Respostas (QA):** Uma API robusta baseada em FastAPI, que recebe perguntas do usuário, realiza busca semântica nos documentos indexados e utiliza um LLM (via Groq) para gerar respostas contextualizadas e precisas, citando as fontes.
- **Expansibilidade e Sistema de Agentes:** O projeto está sendo adaptado para um sistema de agentes, onde cada agente pode ser responsável por um domínio (universidade, empresa, setor), facilitando a expansão e personalização do Lumia para múltiplos contextos e bases de dados.

## Fluxo de Funcionamento

1. **Coleta de Dados:** Scrapers percorrem o site da instituição, extraindo e salvando todo o conteúdo relevante.
2. **Indexação:** O conteúdo é transformado em embeddings e armazenado junto com os metadados.
3. **Pergunta do Usuário:** O usuário faz uma pergunta via interface ou API.
4. **Busca Semântica:** O sistema gera o embedding da pergunta, busca os documentos mais relevantes e monta o contexto.
5. **Resposta com LLM:** O contexto é enviado ao modelo de linguagem, que gera uma resposta clara, citando as fontes.
6. **Regionalismos:** Se a pergunta contiver regionalismos, a resposta é adaptada para o tom/região correspondente.
7. **Sistema de Agentes:** Cada agente pode ser configurado para um domínio específico, tornando o sistema expansível e multi-institucional.

## Diferenciais

- Alta cobertura e eficiência de scraping.
- Busca semântica robusta, independente da forma de escrita.
- Pronto para expansão para múltiplos domínios e empresas.
- Fácil manutenção e atualização de dados.
- Respostas sempre acompanhadas das fontes originais.
- Arquitetura orientada a agentes para máxima flexibilidade e escalabilidade.

---

Este arquivo complementa o `mindlog.md` (registro de ideias e decisões) e o `changelog.md` (registro de mudanças técnicas), servindo como referência geral sobre o propósito, arquitetura e diferenciais do Lumia.
