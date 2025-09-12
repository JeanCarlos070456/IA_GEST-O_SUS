import os, json
import streamlit as st
from dotenv import load_dotenv
import pandas as pd

from config import settings
from utils.rag import search

st.set_page_config(page_title="RAG – Drive QA", layout="wide")
load_dotenv()

st.title("🔎 Assistente de Perguntas & Respostas – Google Drive (MVP)")

with st.sidebar:
    st.header("Config")
    st.caption("Para reindexar, rode no terminal: `python indexer.py`.")
    topk = st.slider("Top-K resultados", 1, 10, 5)
    show_chunks = st.toggle("Mostrar texto dos chunks", value=True)

@st.cache_resource
def load_chunks():
    rows = []
    if os.path.exists(settings.CHUNKS_JSONL):
        with open(settings.CHUNKS_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                rows.append(json.loads(line))
    return rows

chunks = load_chunks()
if not chunks:
    st.warning("Nenhum índice encontrado. Rode `python indexer.py` primeiro.")

query = st.text_input("Faça sua pergunta (ex.: 'Quais metas do projeto X em 2024?')")
btn = st.button("Buscar")

if btn and query.strip():
    with st.spinner("Buscando no índice..."):
        results = search(query, topk, chunks)

    if not results:
        st.info("Nada encontrado. Tente reformular a pergunta.")
    else:
        # Síntese simples: concatenação dos trechos mais relevantes
        answer = "\n\n".join([f"• {r['text'][:500]}..." for r in results])
        st.subheader("Resposta (síntese dos trechos mais relevantes)")
        st.write(answer)

        st.subheader("Fontes")
        df = pd.DataFrame([
            {
                'similaridade': round(r['score'], 3),
                'arquivo': r['doc_title'],
                'chunk': r['chunk_id'],
                'mime': r['mimeType'],
                'caminho_local': r['path']
            } for r in results
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        if show_chunks:
            st.divider()
            for r in results:
                with st.expander(f"{r['doc_title']} — chunk {r['chunk_id']} (score {r['score']:.3f})"):
                    st.write(r['text'])

st.caption("MVP extrativo com embeddings (MiniLM) + FAISS. Próximo passo: integrar LLM para respostas gerativas com citações.")
