# Streamlit RAG – Google Drive (MVP Pronto)

## Como rodar
1. No Google Cloud, habilite *Google Drive API* e *Google Sheets API*.
2. Crie uma *Service Account* e salve o JSON em `credentials/sa-key.json`.
3. Compartilhe a pasta do Drive com o e-mail da Service Account (permissão **Leitor**).
4. Edite o `.env` com `FOLDER_ID` da pasta do Drive.
5. Crie venv e instale dependências:
   ```bash
   pip install -r requirements.txt
   ```
6. Rode o indexador para baixar e indexar os arquivos:
   ```bash
   python indexer.py
   ```
7. Inicie o app:
   ```bash
   streamlit run app.py
   ```

## Estrutura
- `indexer.py` — sincroniza Google Drive e cria índice vetorial (FAISS)
- `app.py` — interface de Perguntas & Respostas no Streamlit
- `utils/drive.py` — cliente Google Drive (PyDrive2)
- `utils/ingest.py` — leitura, limpeza e chunking de documentos
- `utils/rag.py` — embeddings (Sentence-Transformers) e busca FAISS
- `config.py` — parâmetros do projeto
- `data/` — índices e arquivos baixados
- `credentials/` — coloque aqui o `sa-key.json`

## Observações
- Este MVP é **extrativo**: retorna trechos relevantes. Você pode integrar um LLM para respostas gerativas mantendo as citações.
- Ajuste `CHUNK_SIZE` e `CHUNK_OVERLAP` no `.env` conforme o perfil dos documentos.
