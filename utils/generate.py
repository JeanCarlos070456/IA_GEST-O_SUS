import os
from typing import List, Dict

SYSTEM_PROMPT = (
    "Você é um assistente que responde apenas com base nos trechos fornecidos. "
    "Se faltar informação, diga claramente que não encontrou. "
    "Inclua as fontes (arquivo e chunk) ao final."
)

def _format_context(results: List[Dict]) -> str:
    return "\n\n".join([f"[{r['doc_title']} - chunk {r['chunk_id']}] {r['text']}" for r in results])

def _build_prompt(query: str, results: List[Dict]) -> str:
    ctx = _format_context(results)
    return f"{SYSTEM_PROMPT}\n\n# PERGUNTA:\n{query}\n\n# CONTEXTO:\n{ctx}\n\n# RESPOSTA:"

def answer_with_openai(query: str, results: List[Dict]) -> str:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    prompt = _build_prompt(query, results)
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message["content"].strip()

def answer_with_ollama(query: str, results: List[Dict]) -> str:
    import requests
    model = os.getenv("OLLAMA_MODEL", "llama3")
    prompt = _build_prompt(query, results)
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False},
                      timeout=180)
    r.raise_for_status()
    return r.json().get("response", "").strip()
