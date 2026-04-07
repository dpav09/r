import os
from typing import List

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, T5EncoderModel

MODEL_NAME = os.getenv("MODEL_NAME", "ai-forever/FRIDA")
DEVICE = os.getenv("TORCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = os.getenv("MODEL_DTYPE", "float16")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))

if DEVICE.startswith("cuda"):
    torch.backends.cuda.matmul.allow_tf32 = True


def resolve_dtype():
    if not DEVICE.startswith("cuda"):
        return torch.float32
    if MODEL_DTYPE == "bfloat16":
        return torch.bfloat16
    return torch.float16


DTYPE = resolve_dtype()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = T5EncoderModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
).to(DEVICE)
model.eval()

app = FastAPI(title="FRIDA Embedder")


class EmbedRequest(BaseModel):
    texts: List[str]


def _encode(texts: List[str], prefix: str) -> List[List[float]]:
    vectors: List[List[float]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = [f"{prefix}{t}" for t in texts[start : start + BATCH_SIZE]]

        batch = tokenizer(
            batch_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        with torch.inference_mode():
            outputs = model(**batch)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = F.normalize(embeddings, p=2, dim=1)

        vectors.extend(embeddings.float().cpu().tolist())

    return vectors


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/embed/documents")
def embed_documents(req: EmbedRequest):
    return {"vectors": _encode(req.texts, "search_document: ")}


@app.post("/embed/queries")
def embed_queries(req: EmbedRequest):
    return {"vectors": _encode(req.texts, "search_query: ")}
