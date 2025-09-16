# pip install cross-encoder==0.2.3
from typing import List, Dict
from sentence_transformers import CrossEncoder


_model = None

def _load():
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model

def rerank(query: str, results: List[Dict], topn: int = 5) -> List[Dict]:
    model = _load()
    pairs = [(query, r["text"]) for r in results]
    scores = model.predict(pairs)
    for r, s in zip(results, scores):
        r["rerank"] = float(s)
    return sorted(results, key=lambda x: x["rerank"], reverse=True)[:topn]
