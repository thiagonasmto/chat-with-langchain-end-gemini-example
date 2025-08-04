import streamlit as st
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.messages import BaseMessage
from typing import cast, List

# Inicializa o modelo com streaming
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    google_api_key="your_api_key",
    stream=True
)

# Configuração da interface
st.set_page_config(page_title="Chat com IA (Streaming + Memória)", page_icon="🧠", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🧠 Chat com Memória + Streaming</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>O modelo lembra de informações anteriores e responde em tempo real.</p>", unsafe_allow_html=True)
st.divider()

# Inicializa o histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = cast(List[BaseMessage], [])

# Exibe histórico
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Entrada do usuário
user_input = st.chat_input("Digite sua mensagem...")

if user_input:
    # Adiciona mensagem do usuário ao histórico
    user_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(user_msg)
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # Local da resposta
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Streaming de resposta com memória passada
        for chunk in model.stream(st.session_state.messages):
            content = chunk.content if hasattr(chunk, "content") else ""
            full_response += content
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    # Adiciona resposta da IA ao histórico
    ai_msg = AIMessage(content=full_response)
    st.session_state.messages.append(ai_msg)
