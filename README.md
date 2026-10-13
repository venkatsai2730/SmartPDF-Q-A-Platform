# SmartPDF Q&A Platform

**RAG (Retrieval-Augmented Generation)** for PDF question answering.
Upload a PDF → ask questions → get answers taken from the document, with page references and a confidence score.

There is no generative LLM in the loop: answers are extracted from, or assembled out of, the document's own text.

---

## How it works

```
Upload   PDF → text extraction (font-aware headings, tables rebuilt as labelled rows,
               repeated headers/footers removed)
         → 800-char chunks, each with a "Title > Part > Heading" context line
         → BGE-large embeddings → Qdrant (on disk) + BM25 index
         → section index (large documents only)

Ask      question
         → [large documents] pick the best sections first
         → vector search + BM25 → Reciprocal Rank Fusion → candidate pool
         → cross-encoder reranker → top chunks
         → factual question : DeBERTa picks the answer span
           broad question   : whole sentences, in document order, grouped by heading and page
         → confidence = top reranker score; below the gate the answer is "not found"
```

## Technology stack

| Component | Technology | Details |
|---|---|---|
| **Embedder** | BAAI/bge-large-en-v1.5 | 1024-dim, normalized |
| **Vector store** | Qdrant (embedded, on disk) | Exact cosine search, filters on document and chunk; falls back to in-memory exact search; point `QDRANT_URL` at a server for production |
| **Keyword search** | BM25 (rank-bm25) | Fused with vector search by Reciprocal Rank Fusion |
| **Reranker** | BAAI/bge-reranker-base | Cross-encoder; its top score is the confidence |
| **QA reader** | deepset/deberta-v3-large-squad2 | Extractive answers for factual questions |
| **PDF parsing** | PyMuPDF | Font sizes find headings; table cells rebuilt only when no text would be lost |
| **Auth** | JWT (python-jose) + bcrypt | Documents are private to the user who uploaded them |
| **Backend** | FastAPI + Uvicorn | |
| **Frontend** | React 18 + Axios | |
| **Deployment** | Docker Compose | Backend + frontend; a volume keeps documents across restarts |

## Large documents

Above `TWO_STAGE_MIN_CHUNKS` (default 150) the document is also split into sections (heading to heading, at most 12 chunks each). A question first picks its best sections, then the chunk search runs inside them, plus a few global hits as a safety net. Smaller documents are searched whole.

## Folder structure

```
SmartPDF-Q-A-Platform/
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── server.py               # FastAPI app, extraction, retrieval pipeline, storage
│   ├── tests/test_retrieval.py # unit tests (no ML models needed)
│   └── data/                   # created at runtime: documents, vectors, users (git-ignored)
└── frontend/
    ├── Dockerfile
    ├── package.json
    └── src/                    # App.js, App.css, index.js
```

## Setup & run

### Docker (recommended)

```bash
docker compose up --build
# Backend:  http://localhost:8100
# Frontend: http://localhost:3100
# First start downloads ~4 GB of models; the backend answers 503 "models still loading" until they are ready
```

### Local development

```bash
cd backend && pip install -r requirements.txt
uvicorn server:app --reload --host 0.0.0.0 --port 8100     # backend

cd frontend && npm install && npm start                     # frontend (port 3100)
```

### Tests

```bash
cd backend && python -m pytest tests
```

### Default credentials (change before any real use)

| Username | Password |
|---|---|
| `admin` | `admin123` |
| `demo` | `demo123` |

Set `JWT_SECRET_KEY` to your own secret.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `JWT_SECRET_KEY` | built-in dev key | Signing key for login tokens: always override |
| `DATA_DIR` | `backend/data` | Where documents, vectors and registered users are saved |
| `PERSIST` | `1` | `0` keeps everything in memory only |
| `VECTOR_BACKEND` | `qdrant` | `numpy` = exact in-memory search (still saved to disk when `PERSIST=1`) |
| `QDRANT_URL` | empty | Use a Qdrant server instead of the embedded on-disk store |
| `TWO_STAGE` | `1` | Section-first search for large documents |
| `TWO_STAGE_MIN_CHUNKS` | `150` | Documents below this size are searched whole |
| `TOP_SECTIONS` | `6` | Sections searched per question |
| `MIN_RERANK_SCORE` | `0.10` | "Not found" gate for factual questions |
| `MIN_RERANK_SCORE_BROAD` | `0.03` | Gate for broad questions (the reranker also sees the context line, so scores run lower) |
| `RERANK_WITH_CONTEXT` | `1` | Let the reranker see the context line for broad questions |

Both gates were tuned on a handful of test documents. Re-tune them on your own questions.

## API endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/login` | no | Get a JWT (OAuth2 form) |
| `POST` | `/register` | no | Create an account and get a JWT |
| `POST` | `/upload/` | yes | Upload a PDF for indexing |
| `POST` | `/ask/` | yes | Ask a question about one of your documents |
| `GET` | `/doc-info/{doc_id}` | yes | Sections, tags and filters of your document |
| `GET` | `/health/` | no | Server health and model status |
| `GET` | `/stats/` | yes | Metrics and your own documents |

## Measured results

Each set is 22–31 factual, 4–8 broad and 9–12 unanswerable questions, written by us with answers checked against the extracted text. One run per document, on a CPU-only laptop. Treat them as indicative, not as benchmarks.

| Document | Right chunk in the final top 5 | "Near-miss" questions refused* | Broad answers cover the expected points |
|---|---|---|---|
| Design summary, 6 pages (+60 noise pages) | 84–100% | high | 77% |
| Llama 2 paper, 77 pages | 93% | 50% (67% with two-stage) | 49–60% |
| DPDP Act 2023, 21 pages | 86% (95% with two-stage) | 0% | 58% |

\*Questions that sound answerable but are not in the document, for example asking for the GDPR fine in a document about India's DPDP Act.

### Known limitations

- **Plausible questions the document does not answer can still get an answer.** Totally unrelated questions are refused reliably, but a question that shares topic words with the document (a GDPR fine against the DPDP Act) can score high and be answered from the wrong text.
- **Broad questions on large documents cover only part of the topic**, because the facts are spread over many pages and only the top few chunks are used.
- **Short definition-style questions can be refused** when the reranker scores the right sentence low.
- **Tables with merged cells or split headers** are left as plain lines rather than rebuilt; the Act's Schedule of penalties is an example.
- **Scanned (image-only) PDFs are not supported**; there is no OCR.
- Registered users are saved, but the built-in `admin` and `demo` accounts and the dev signing key are for local use only.
