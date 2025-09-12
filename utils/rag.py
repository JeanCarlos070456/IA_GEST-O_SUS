import pickle
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss

from config import settings

_model = None
_index = None
_meta = None

def load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.MODEL_NAME)
    return _model

def build_or_load_faiss(embeddings: np.ndarray, ids: List[int]):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    # normaliza vetores (cosine similarity via dot product)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    with open(settings.FAISS_META, 'wb') as f:
        pickle.dump({'ids': ids, 'dim': dim}, f)
    faiss.write_index(index, settings.FAISS_BIN)
    return index

def load_index() -> Tuple[faiss.Index, Dict]:
    global _index, _meta
    if _index is None:
        with open(settings.FAISS_META, 'rb') as f:
            _meta = pickle.load(f)
        _index = faiss.read_index(settings.FAISS_BIN)
    return _index, _meta

def encode_texts(texts: List[str]) -> np.ndarray:
    model = load_model()
    embs = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype('float32')

def search(query: str, k: int, chunks: List[Dict]) -> List[Dict]:
    index, meta = load_index()
    qv = encode_texts([query])
    D, I = index.search(qv, k)
    results = []
    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue
        row = chunks[meta['ids'][idx]].copy()
        row['score'] = float(D[0][rank])
        results.append(row)
    return results
