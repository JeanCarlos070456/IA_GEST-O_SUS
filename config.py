from dataclasses import dataclass
import os

@dataclass
class Settings:
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    RAW_DIR: str = os.path.join(DATA_DIR, "raw")
    CREDENTIALS_JSON: str = os.getenv("CREDENTIALS_JSON", "credentials/sa-key.json")
    DRIVE_FOLDER_ID: str = os.getenv("FOLDER_ID", "")
    MODEL_NAME: str = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 900))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 150))
    MAX_FILES: int = int(os.getenv("MAX_FILES", 2000))

    FAISS_BIN: str = os.path.join(DATA_DIR, "faiss_index.bin")
    FAISS_META: str = os.path.join(DATA_DIR, "faiss_index.pkl")
    CHUNKS_JSONL: str = os.path.join(DATA_DIR, "chunks.jsonl")

settings = Settings()

os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.RAW_DIR, exist_ok=True)
