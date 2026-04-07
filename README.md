# contracts-rag-gpu

GPU-first MVP для legal RAG:

- Qdrant
- FRIDA embeddings
- BM25 / sparse search в Qdrant
- Hybrid retrieval: dense + BM25 + RRF
- `BAAI/bge-reranker-v2-m3`
- LLM answer layer через vLLM OpenAI-compatible API

## Что внутри

- `services/embedder` — отдельный сервис embeddings на FRIDA
- `services/reranker` — отдельный сервис reranking
- `services/api` — индексация, hybrid search и answer-layer orchestration
- `qdrant` — векторная БД
- `llm` — локальный OpenAI-compatible сервер на vLLM

## 1. Подготовка хоста

На хосте должны быть:

- актуальный NVIDIA driver
- Docker Engine + Docker Compose plugin
- NVIDIA Container Toolkit

Проверьте GPU на хосте:

```bash
nvidia-smi
```

После установки NVIDIA Container Toolkit обычно нужно выполнить:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

И проверить доступ Docker к GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
```

## 2. Настройка env

Файл `.env` создайте копированием из `.env.example`:

```bash
cp .env.example .env
```

При желании обновите:

- `LLM_MODEL`
- `HF_TOKEN`
- `EMBEDDER_BATCH_SIZE`
- `LLM_MAX_TOKENS`

## 3. Запуск

```bash
docker compose up -d --build
docker compose exec api python create_collection.py
```

Проверка сервисов:

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/v1/models -H "Authorization: Bearer local-dev-key"
```

## 4. Индексация одного договора

```bash
curl -X POST http://localhost:8000/index/document \
  -H "Content-Type: application/json" \
  -d '{
    "contract_id": "nda_001",
    "contract_type": "nda",
    "counterparty": "ООО Ромашка",
    "source_file": "nda_001.txt",
    "text": "РАЗДЕЛ 1. ПРЕДМЕТ ДОГОВОРА\n\n1.1 ..."
  }'
```

## 5. Поиск

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Какая ответственность предусмотрена за разглашение конфиденциальной информации?",
    "top_k": 5
  }'
```

## 6. Answer layer

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Какая ответственность предусмотрена за разглашение конфиденциальной информации?",
    "retrieval_top_k": 6
  }'
```

## 7. Практические замечания

- На одной RTX 5090 все три GPU-сервиса (`embedder`, `reranker`, `llm`) делят одну видеокарту.
- В `llm` уже выставлен более осторожный режим:
  - `--gpu-memory-utilization 0.55`
  - `--max-model-len 16384`
- Если словите OOM, сначала уменьшайте:
  - `EMBEDDER_BATCH_SIZE`
  - `LLM_CONTEXT_CHUNKS`
  - `LLM_MAX_TOKENS`
  - `--gpu-memory-utilization` у `llm`
- Если всё стабильно, потом постепенно поднимайте лимиты.

## 8. Структура payload чанка

Каждый чанк хранит:

- `contract_id`
- `contract_type`
- `counterparty`
- `source_file`
- `chunk_no`
- `chunk_text`

Дальше можно добавить:

- `section_path`
- `effective_date`
- `clause_no`
- `governing_law`
- `currency`
- `party_role`
