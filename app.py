# app.py – RAG Drive (versão estendida)
import os, json, time
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from importlib import reload

from config import settings
from utils.rag import search

st.set_page_config(page_title="RAG – Drive QA", layout="wide")
load_dotenv()

st.title("🔎 Assistente de Perguntas & Respostas – Google Drive")

# ======================
# Sidebar: controles
# ======================
with st.sidebar:
    st.header("Configurações")
    st.caption("Você pode reindexar aqui mesmo. Se algo falhar, rode `python indexer.py` pelo terminal.")

    # Botão de reindex no app
    if st.button("🔁 Reindexar agora"):
        with st.spinner("Reindexando a(s) pasta(s) do Drive..."):
            try:
                import indexer as indexer_mod
                reload(indexer_mod)
                indexer_mod.main()
                st.success("Reindex concluído com sucesso!")
                # Limpa caches para recarregar o novo índice
                st.cache_resource.clear()
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Falha ao reindexar: {e}")

    topk = st.slider("Top-K resultados", 1, 15, 5)
    show_chunks = st.toggle("Mostrar texto dos chunks", value=True)

    # Filtros por "tipo" (heurística pelo caminho/ extensão)
    mime_filter = st.multiselect(
        "Filtrar por tipo de arquivo",
        ["pdf", "docx", "pptx", "xlsx", "xls", "csv", "txt", "html", "htm"],
        []
    )

    # Re-ranking (opcional)
    use_rerank = st.toggle("Re-ranking (Cross-Encoder)", value=False, help="Reordena o Top-K com um modelo de reclassificação.")

    # LLM (opcional)
    use_llm = st.toggle("Usar síntese com LLM (RAG)", value=False, help="Gera resposta com base nos trechos recuperados.")
    llm_backend = None
    if use_llm:
        llm_backend = st.selectbox("Backend LLM", ["OpenAI", "Ollama"], index=1)

# ======================
# Cache de chunks e métricas
# ======================
@st.cache_resource
def load_chunks():
    rows = []
    if os.path.exists(settings.CHUNKS_JSONL):
        with open(settings.CHUNKS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows

@st.cache_data
def idx_stats():
    path = settings.CHUNKS_JSONL
    total = 0
    mtime = None
    if os.path.exists(path):
        total = sum(1 for _ in open(path, "r", encoding="utf-8"))
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
    return {"total_chunks": total, "last_index_time": mtime}

chunks = load_chunks()
stats = idx_stats()

if not chunks:
    st.warning("Nenhum índice encontrado. Clique em **Reindexar agora** ou rode `python indexer.py` primeiro.")
else:
    c1, c2 = st.columns(2)
    c1.metric("Chunks indexados", stats["total_chunks"])
    c2.metric("Última indexação", stats["last_index_time"] or "—")

# ======================
# Busca
# ======================
query = st.text_input("Faça sua pergunta (ex.: 'Quais metas do projeto X em 2024?')")
btn = st.button("Buscar")

if btn and query.strip():
    with st.spinner("Buscando no índice..."):
        results = search(query, topk, chunks)

    # Filtro por tipo de arquivo (extensão no caminho)
    if mime_filter:
        exts = tuple(ext.lower() for ext in mime_filter)
        results = [r for r in results if r.get("path", "").lower().endswith(exts)]

    # Re-ranking (opcional)
    if use_rerank and results:
        try:
            from utils.rerank import rerank  # requer arquivo opcional
            results = rerank(query, results, topn=topk)
        except Exception as e:
            st.info(f"Re-ranking desabilitado (módulo indisponível): {e}")

    if not results:
        st.info("Nada encontrado. Tente reformular a pergunta, ajustar Top-K ou filtros.")
    else:
        # ======================
        # Resposta
        # ======================
        if use_llm:
            try:
                from utils.generate import answer_with_openai, answer_with_ollama  # arquivo opcional
                if llm_backend == "OpenAI":
                    final_answer = answer_with_openai(query, results)
                else:
                    final_answer = answer_with_ollama(query, results)

                st.subheader("Resposta (LLM + citações)")
                st.write(final_answer)
            except Exception as e:
                st.warning(f"Falha ao sintetizar com LLM: {e}")
                # fallback extrativo
                answer = "\n\n".join([f"• {r['text'][:500]}..." for r in results])
                st.subheader("Resposta (extrativa – fallback)")
                st.write(answer)
        else:
            # extrativo (padrão)
            answer = "\n\n".join([f"• {r['text'][:500]}..." for r in results])
            st.subheader("Resposta (síntese dos trechos mais relevantes)")
            st.write(answer)

        # ======================
        # Fontes
        # ======================
        st.subheader("Fontes")
        df = pd.DataFrame([
            {
                "similaridade": round(r.get("score", 0.0), 3),
                "arquivo": r.get("doc_title"),
                "chunk": r.get("chunk_id"),
                "mime": r.get("mimeType"),
                "caminho_local": r.get("path"),
            } for r in results
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ======================
        # Chunks (opcional)
        # ======================
        if show_chunks:
            st.divider()
            for r in results:
                title = r.get("doc_title", "Documento")
                chunk_id = r.get("chunk_id", 0)
                score = r.get("score", 0.0)
                with st.expander(f"{title} — chunk {chunk_id} (score {score:.3f})"):
                    st.write(r.get("text", ""))

st.caption("RAG extrativo com embeddings (MiniLM) + FAISS. Opções: Re-index, filtros, re-ranking e síntese LLM.")

