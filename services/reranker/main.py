import os
from typing import List

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from FlagEmbedding import FlagReranker

MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-reranker-v2-m3")
USE_FP16 = os.getenv("USE_FP16", "true").lower() in {"1", "true", "yes"}
MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "1024"))

reranker = FlagReranker(
    MODEL_NAME,
    use_fp16=USE_FP16 and torch.cuda.is_available(),
)

app = FastAPI(title="BGE Reranker")


class RerankRequest(BaseModel):
    query: str
    passages: List[str]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "cuda_available": torch.cuda.is_available(),
        "use_fp16": USE_FP16 and torch.cuda.is_available(),
        "max_length": MAX_LENGTH,
    }


@app.post("/rerank")
def rerank(req: RerankRequest):
    if not req.passages:
        return {"scores": []}

    pairs = [[req.query, p] for p in req.passages]
    scores = reranker.compute_score(
        pairs,
        normalize=True,
        max_length=MAX_LENGTH,
    )

    if isinstance(scores, float):
        scores = [scores]

    return {"scores": list(scores)}
