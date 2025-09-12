import os, json
from dotenv import load_dotenv
from tqdm import tqdm

from config import settings
from utils.drive import make_drive_client, list_files_in_folder, download_file
from utils.ingest import read_text_from_file, chunk_text, write_jsonl
from utils.rag import encode_texts, build_or_load_faiss

def main():
    load_dotenv()
    if not settings.DRIVE_FOLDER_ID:
        raise RuntimeError("Defina FOLDER_ID no .env")

    drive = make_drive_client()
    files = list_files_in_folder(drive, settings.DRIVE_FOLDER_ID)

    rows = []
    for f in tqdm(files[:settings.MAX_FILES], desc="Baixando & extraindo"):
        title = f['title']
        mime = f['mimeType']
        fid = f['id']

        # Define nome destino por id + extensão inferida
        ext_map = {
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
            'application/pdf': '.pdf',
            'text/plain': '.txt',
            'text/csv': '.csv',
            'text/html': '.html',
        }
        ext = os.path.splitext(title)[1].lower()
        if not ext:
            ext = ext_map.get(mime, '.bin')
        dst = os.path.join(settings.RAW_DIR, f"{fid}{ext}")

        if not os.path.exists(dst):
            download_file(drive, f, dst)

        # Extrai texto
        try:
            text, meta = read_text_from_file(dst)
        except Exception as e:
            print("Falha ao ler:", dst, e)
            continue

        # Chunking
        chunks = chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        for i, c in enumerate(chunks):
            rows.append({
                'doc_id': fid,
                'doc_title': title,
                'chunk_id': i,
                'text': c,
                'mimeType': mime,
                'path': dst,
                'drive_id': fid,
            })

    # Salva JSONL
    write_jsonl(settings.CHUNKS_JSONL, rows)

    # Embeddings & FAISS
    texts = [r['text'] for r in rows]
    if texts:
        embs = encode_texts(texts)
        ids = list(range(len(rows)))
        build_or_load_faiss(embs, ids)
        print(f"Indexados {len(rows)} chunks de {len(files)} arquivos.")
    else:
        print("Nenhum texto indexado. Verifique os formatos ou permissões da pasta.")

if __name__ == "__main__":
    main()
