import streamlit as st
import requests
import time
import uuid # Usar UUID para IDs mais robustos

# --- Configuration --- A API URL é configurada aqui
# Aponta para o serviço backend local
API_URL = "http://localhost:8000/ask"
PAGE_TITLE = "LumIA Chat UFPB"
# LOGO_PLACEHOLDER = "🎓 LumIA" # Removido

# --- Initialization ---
# Inicializa o estado da sessão se não existir
if "chats" not in st.session_state:
    # Estrutura: {chat_id: {"title": str, "messages": [{"role": str, "content": str, "message_id": str}]}}
    st.session_state.chats = {}
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

# --- Helper Functions ---
def generate_chat_id() -> str:
    "Gera um ID único para um novo chat."
    return f"chat_{uuid.uuid4()}"

def generate_message_id(role: str) -> str:
    "Gera um ID único para uma mensagem."
    return f"{role}_{uuid.uuid4()}"

def get_api_response(chat_id: str, user_prompt: str):
    "Envia a pergunta para a API e atualiza o estado do chat."
    loading_id = generate_message_id("loading")
    st.session_state.chats[chat_id]["messages"].append(
        {"role": "assistant", "content": "...", "message_id": loading_id}
    )
    # Força um rerun para mostrar a mensagem de "loading" imediatamente.
    # A API call acontece DEPOIS deste rerun.
    st.session_state.prompt_to_process = {"chat_id": chat_id, "prompt": user_prompt, "loading_id": loading_id}
    st.rerun()

def process_api_call():
    "Processa a chamada à API que foi agendada no rerun anterior."
    if "prompt_to_process" in st.session_state and st.session_state.prompt_to_process:
        call_data = st.session_state.prompt_to_process
        chat_id = call_data["chat_id"]
        user_prompt = call_data["prompt"]
        loading_id = call_data["loading_id"]

        # Limpa o agendamento para não processar novamente
        st.session_state.prompt_to_process = None

        try:
            response = requests.post(API_URL, json={"text": user_prompt}, timeout=60) # Timeout maior
            response.raise_for_status()
            data = response.json()
            answer = data.get("answer", "Desculpe, não recebi uma resposta válida.")
            final_message = {"role": "assistant", "content": answer, "message_id": generate_message_id("assistant")}

        except requests.exceptions.Timeout:
             error_message = "Erro: Tempo limite excedido ao conectar com a LumIA."
             final_message = {"role": "assistant", "content": error_message, "message_id": generate_message_id("error")}
        except requests.exceptions.RequestException as e:
            print(f"API Request Error: {e}")
            error_message = f"Erro de conexão com a API: {e}"
            final_message = {"role": "assistant", "content": error_message, "message_id": generate_message_id("error")}

        # Encontra e atualiza a mensagem de loading
        updated = False
        for i, msg in enumerate(st.session_state.chats[chat_id]["messages"]):
            if msg.get("message_id") == loading_id:
                st.session_state.chats[chat_id]["messages"][i] = final_message
                updated = True
                break
        if not updated:
             print(f"Warning: Loading message {loading_id} not found to update.")
             # Adiciona a mensagem final se a de loading sumiu por algum motivo
             st.session_state.chats[chat_id]["messages"].append(final_message)

        st.rerun()

# --- Process API Call if scheduled ---
process_api_call()

# --- UI Rendering ---
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# --- Sidebar ---
with st.sidebar:
    # Criar colunas para centralizar a imagem
    # Ajuste as proporções [1, 2, 1] se necessário para melhor centralização
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2: # Coloca a imagem na coluna central
        st.image("assets/logo.png", width=150) # Exibe a imagem

    st.divider() # Adiciona um divisor abaixo da imagem/logo

    if st.button("➕ Novo Chat", use_container_width=True):
        new_id = generate_chat_id()
        chat_num = len(st.session_state.chats) + 1
        st.session_state.chats[new_id] = {
            "title": f"Novo Chat {chat_num}",
            "messages": []
        }
        st.session_state.active_chat_id = new_id
        st.rerun()

    st.divider()
    st.write("**Chats Salvos**")

    # Ordena chats por tempo de criação (implícito no ID baseado em UUIDv1 ou time)
    # Se usar UUIDv4, a ordem é aleatória. Poderia adicionar um timestamp de criação ao chat.
    chat_ids = list(st.session_state.chats.keys())
    # chat_ids.sort(key=lambda x: st.session_state.chats[x].get('creation_time', 0), reverse=True) # Exemplo se tivesse timestamp
    chat_ids.reverse() # Simplesmente inverte a ordem de inserção (mais recentes primeiro)

    if not chat_ids:
        st.caption("Nenhum chat iniciado.")

    for chat_id in chat_ids:
        chat_title = st.session_state.chats[chat_id].get("title", chat_id)
        button_key = f"select_{chat_id}"
        is_active = chat_id == st.session_state.active_chat_id
        # Usa um truque com markdown para "simular" botão ativo ou st.radio
        label = f"**{chat_title}**" if is_active else chat_title
        if st.button(label, key=button_key, use_container_width=True):
            st.session_state.active_chat_id = chat_id
            st.rerun()

# --- Main Chat Area ---
if st.session_state.active_chat_id is None:
    st.info("Selecione um chat na barra lateral ou crie um novo para começar.")
else:
    active_chat_id = st.session_state.active_chat_id
    if active_chat_id not in st.session_state.chats:
        # Segurança: se o chat ativo foi deletado ou estado corrompido
        st.error("Chat ativo não encontrado. Selecione outro chat.")
        st.session_state.active_chat_id = None
    else:
        active_chat = st.session_state.chats[active_chat_id]
        st.header(active_chat["title"])

        # Display messages
        for message in active_chat["messages"]:
            role = message["role"]
            # Garante que role seja 'user' ou 'assistant' para st.chat_message
            display_role = "user" if role == "user" else "assistant"
            with st.chat_message(display_role):
                st.markdown(message["content"])

        # Chat input - Usar key única para cada chat garante que o estado do input resete ao mudar de chat
        prompt_key = f"input_{active_chat_id}"
        if prompt := st.chat_input("Digite sua mensagem...", key=prompt_key):
            # Adiciona mensagem do usuário ao estado
            st.session_state.chats[active_chat_id]["messages"].append(
                {"role": "user", "content": prompt, "message_id": generate_message_id("user")}
            )
            # Agenda a chamada da API para o próximo rerun
            get_api_response(active_chat_id, prompt)