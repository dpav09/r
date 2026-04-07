import os
import re
from typing import Any, Dict, List, Tuple

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "contracts_chunks")

EMBEDDER_URL = os.getenv("EMBEDDER_URL", "http://embedder:8000")
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8000")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://llm:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-14B-AWQ")
LLM_API_KEY = os.getenv("LLM_API_KEY", "local-dev-key")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "700"))
LLM_CONTEXT_CHUNKS = int(os.getenv("LLM_CONTEXT_CHUNKS", "6"))

PREFETCH_LIMIT = int(os.getenv("PREFETCH_LIMIT", "60"))
FUSED_LIMIT = int(os.getenv("FUSED_LIMIT", "20"))
INDEX_BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "64"))

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))

SYSTEM_PROMPT = """Ты помощник по корпоративным договорам.
Отвечай только по предоставленным фрагментам договора.
Если данных недостаточно, прямо скажи это.
Каждое утверждение в ответе помечай ссылками на источники в формате [S1], [S2].
Не придумывай статьи, пункты, даты, суммы, штрафы и обязательства.
Если фрагменты противоречат друг другу, покажи оба варианта и укажи источники.
Сначала дай короткий прямой ответ, затем краткое пояснение по пунктам.
"""

HEADING_RE = re.compile(
    r"(?im)^(раздел|глава|статья|пункт|подпункт|приложение)\s+[\w\.\-IVXА-Яа-я]+.*$"
)

app = FastAPI(title="Legal Hybrid RAG API")


class IndexDocumentRequest(BaseModel):
    contract_id: str
    text: str
    contract_type: str | None = None
    counterparty: str | None = None
    source_file: str | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)


class AnswerRequest(BaseModel):
    query: str
    retrieval_top_k: int = Field(default=6, ge=1, le=50)


def _post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    headers: Dict[str, str] | None = None,
    timeout: int = REQUEST_TIMEOUT,
) -> Dict[str, Any]:
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"HTTP error for {url}: {resp.status_code} {resp.text}") from e
    return resp.json()


def _put_json(
    url: str,
    payload: Dict[str, Any],
    *,
    headers: Dict[str, str] | None = None,
    timeout: int = REQUEST_TIMEOUT,
) -> Dict[str, Any]:
    resp = requests.put(url, json=payload, headers=headers, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"HTTP error for {url}: {resp.status_code} {resp.text}") from e
    return resp.json()


def simple_legal_chunker(text: str, max_chars: int = 1800, overlap: int = 200) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        is_heading = bool(HEADING_RE.match(para))

        if is_heading and current:
            chunks.append(current.strip())
            current = para
            continue

        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
                tail = current[-overlap:] if overlap and len(current) > overlap else ""
                current = f"{tail}\n\n{para}".strip()
            else:
                start = 0
                while start < len(para):
                    end = start + max_chars
                    chunk = para[start:end].strip()
                    if chunk:
                        chunks.append(chunk)
                    if end >= len(para):
                        break
                    start = max(0, end - overlap)
                current = ""

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]


def estimate_avg_len(texts: List[str]) -> float:
    lengths = [max(1, len(t.split())) for t in texts]
    return sum(lengths) / max(1, len(lengths))


def embed_documents(texts: List[str]) -> List[List[float]]:
    data = _post_json(
        f"{EMBEDDER_URL}/embed/documents",
        {"texts": texts},
    )
    return data["vectors"]


def embed_queries(texts: List[str]) -> List[List[float]]:
    data = _post_json(
        f"{EMBEDDER_URL}/embed/queries",
        {"texts": texts},
    )
    return data["vectors"]


def rerank(query: str, passages: List[str]) -> List[float]:
    if not passages:
        return []

    data = _post_json(
        f"{RERANKER_URL}/rerank",
        {"query": query, "passages": passages},
    )
    return data["scores"]


def qdrant_upsert(points: List[Dict[str, Any]]) -> None:
    for i in range(0, len(points), INDEX_BATCH_SIZE):
        batch = points[i : i + INDEX_BATCH_SIZE]
        _put_json(
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points?wait=true",
            {"points": batch},
        )


def qdrant_hybrid_query(query_text: str, dense_query: List[float]) -> List[Dict[str, Any]]:
    payload = {
        "prefetch": [
            {
                "query": dense_query,
                "using": "frida_dense",
                "limit": PREFETCH_LIMIT,
            },
            {
                "query": {
                    "text": query_text,
                    "model": "qdrant/bm25",
                    "options": {
                        "language": "none",
                        "tokenizer": "multilingual",
                    },
                },
                "using": "bm25_sparse",
                "limit": PREFETCH_LIMIT,
            },
        ],
        "query": {
            "fusion": "rrf",
        },
        "limit": FUSED_LIMIT,
        "with_payload": True,
    }

    data = _post_json(
        f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/query",
        payload,
    )

    raw_result = data.get("result", {})
    if isinstance(raw_result, dict) and "points" in raw_result:
        return raw_result["points"]
    if isinstance(raw_result, list):
        return raw_result
    return []


def build_points(
    *,
    contract_id: str,
    contract_type: str | None,
    counterparty: str | None,
    source_file: str | None,
    chunks: List[str],
) -> List[Dict[str, Any]]:
    dense_vectors = embed_documents(chunks)
    avg_len = estimate_avg_len(chunks)

    points: List[Dict[str, Any]] = []
    for i, (chunk_text, dense) in enumerate(zip(chunks, dense_vectors), start=1):
        points.append(
            {
                "id": f"{contract_id}-{i}",
                "vector": {
                    "frida_dense": dense,
                    "bm25_sparse": {
                        "text": chunk_text,
                        "model": "qdrant/bm25",
                        "options": {
                            "avg_len": avg_len,
                            "language": "none",
                            "tokenizer": "multilingual",
                        },
                    },
                },
                "payload": {
                    "contract_id": contract_id,
                    "contract_type": contract_type,
                    "counterparty": counterparty,
                    "source_file": source_file,
                    "chunk_no": i,
                    "chunk_text": chunk_text,
                },
            }
        )

    return points


def run_search(query_text: str, top_k: int) -> List[Dict[str, Any]]:
    dense_query = embed_queries([query_text])[0]
    points = qdrant_hybrid_query(query_text, dense_query)

    if not points:
        return []

    passages = [p["payload"]["chunk_text"] for p in points]
    rerank_scores = rerank(query_text, passages)

    results: List[Dict[str, Any]] = []
    for point, rerank_score in zip(points, rerank_scores):
        results.append(
            {
                "id": point.get("id"),
                "rrf_score": point.get("score"),
                "rerank_score": float(rerank_score),
                "payload": point.get("payload", {}),
            }
        )

    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results[:top_k]


def build_context(results: List[Dict[str, Any]], max_chunks: int) -> Tuple[str, List[Dict[str, Any]]]:
    blocks: List[str] = []
    sources: List[Dict[str, Any]] = []

    for idx, item in enumerate(results[:max_chunks], start=1):
        source_id = f"S{idx}"
        payload = item["payload"]

        blocks.append(
            f"[{source_id}]\n"
            f"contract_id: {payload.get('contract_id')}\n"
            f"contract_type: {payload.get('contract_type')}\n"
            f"counterparty: {payload.get('counterparty')}\n"
            f"source_file: {payload.get('source_file')}\n"
            f"chunk_no: {payload.get('chunk_no')}\n"
            f"text:\n{payload.get('chunk_text')}"
        )

        sources.append(
            {
                "source_id": source_id,
                "contract_id": payload.get("contract_id"),
                "source_file": payload.get("source_file"),
                "chunk_no": payload.get("chunk_no"),
                "rerank_score": item.get("rerank_score"),
            }
        )

    return "\n\n---\n\n".join(blocks), sources


def call_llm(messages: List[Dict[str, str]]) -> str:
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }

    data = _post_json(
        f"{LLM_BASE_URL}/chat/completions",
        payload,
        headers=headers,
    )

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected LLM response: {data}") from e


@app.get("/health")
def health():
    return {
        "status": "ok",
        "qdrant_url": QDRANT_URL,
        "collection": QDRANT_COLLECTION,
        "embedder_url": EMBEDDER_URL,
        "reranker_url": RERANKER_URL,
        "llm_base_url": LLM_BASE_URL,
        "llm_model": LLM_MODEL,
    }


@app.post("/index/document")
def index_document(req: IndexDocumentRequest):
    try:
        chunks = simple_legal_chunker(req.text)
        if not chunks:
            return {
                "indexed_chunks": 0,
                "contract_id": req.contract_id,
                "message": "Пустой текст после нормализации/чанкинга.",
            }

        points = build_points(
            contract_id=req.contract_id,
            contract_type=req.contract_type,
            counterparty=req.counterparty,
            source_file=req.source_file,
            chunks=chunks,
        )
        qdrant_upsert(points)

        return {
            "indexed_chunks": len(points),
            "contract_id": req.contract_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/search")
def search(req: SearchRequest):
    try:
        results = run_search(req.query, req.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/answer")
def answer(req: AnswerRequest):
    try:
        retrieval_top_k = max(req.retrieval_top_k, LLM_CONTEXT_CHUNKS)
        results = run_search(req.query, retrieval_top_k)

        if not results:
            return {
                "answer": "Ничего релевантного не найдено в базе договоров.",
                "sources": [],
                "retrieved_chunks": 0,
            }

        context_text, sources = build_context(results, LLM_CONTEXT_CHUNKS)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Вопрос пользователя:\n{req.query}\n\n"
                    f"Фрагменты договора:\n{context_text}\n\n"
                    "Сформируй ответ только на основе этих фрагментов."
                ),
            },
        ]

        answer_text = call_llm(messages)

        return {
            "answer": answer_text,
            "sources": sources,
            "retrieved_chunks": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
