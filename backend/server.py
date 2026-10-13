"""
SmartPDF Q&A Platform — Production Backend Server
FastAPI + Staged Hybrid Filtering Pipeline + JWT Authentication

Pipeline (3-Stage Hybrid Filtering):
  Stage 1 - Pre-filter (indexed metadata):
    page_range, section/department, access_level → selective subset
  Stage 2 - Hybrid Search:
    BGE-large-en-v1.5 → vector search (dense) + BM25 (sparse) on filtered subset,
    merged with Reciprocal Rank Fusion
  Stage 3 - Post-filter + Rerank:
    word_count, tags, content quality → BGE-reranker-base → DeBERTa QA
    (top reranker score = confidence; below MIN_RERANK_SCORE → "not found")

Models:
  - Embedder:  BAAI/bge-large-en-v1.5 (1024-dim, normalize_embeddings=True)
  - Reranker:  BAAI/bge-reranker-base (CrossEncoder)
  - QA Reader: deepset/deberta-v3-large-squad2 (EM ~86% on SQuAD2.0)

Run:
  uvicorn server:app --reload --host 0.0.0.0 --port 8100
"""

import json
import uuid
import time
import logging
import tempfile
import os
import re
import unicodedata
import bisect
import asyncio
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import numpy as np
import bcrypt
from rank_bm25 import BM25Okapi
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from jose import JWTError, jwt


# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="SmartPDF Q&A API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dedicated thread pool for CPU-bound operations
_cpu_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cpu-bound")

# Max upload size: 50MB
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

# Retrieval tuning
DENSE_TOP_K = 30        # candidates taken from the vector search
BM25_TOP_K = 30         # candidates taken from BM25
FUSED_TOP_K = 20        # candidates kept after Reciprocal Rank Fusion
# Broad "summarize / explain / list" questions: the reranker also sees each chunk's context line
# (title > part > heading), which surfaces every section of a topic. Factual questions are always
# reranked on the plain chunk text. (Search itself always uses the context line.)
RERANK_WITH_CONTEXT = os.getenv("RERANK_WITH_CONTEXT", "1") == "1"
# Two-stage search for large documents: find the best sections first, then search chunks inside them
# (plus a few global hits, in case the right section was missed).
TWO_STAGE = os.getenv("TWO_STAGE", "1") == "1"
TWO_STAGE_MIN_CHUNKS = int(os.getenv("TWO_STAGE_MIN_CHUNKS", "150"))   # smaller documents are searched whole
TOP_SECTIONS = int(os.getenv("TOP_SECTIONS", "6"))
SECTION_MAX_CHUNKS = 12
SECTION_EMBED_CHARS = 1500
GLOBAL_EXTRA = 8
GUARANTEED_PER_RETRIEVER = 15  # each retriever's own best, added to the reranker pool
CANDIDATE_CAP = 30      # max chunks the reranker scores per question
RRF_K = 60              # RRF damping constant (Cormack et al. 2009)
# Below this top reranker score (0-1) the answer is "not found". Plain-text reranking (factual
# questions) scores correct answers high and unrelated ones very low: gate 0.10. With the context
# line in the input (broad questions) scores run about 10x lower, so that gate is 0.03.
# Both were set from one 51-question test on one document: re-tune on your own questions.
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.10"))
MIN_RERANK_SCORE_BROAD = float(os.getenv("MIN_RERANK_SCORE_BROAD", "0.03"))
RETRIEVAL_PIPELINE_LABEL = "Stage1:PreFilter → Stage2:Vector+BM25 (RRF) → Stage3:PostFilter+Rerank"


# ---------------------------------------------------------------------------
# Authentication — JWT + Password Hashing
# ---------------------------------------------------------------------------

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "smartpdf-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def hash_password_sync(plain_password: str) -> str:
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password_sync(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


async def hash_password_async(plain_password: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_cpu_executor, hash_password_sync, plain_password)


async def verify_password_async(plain_password: str, hashed_password: str) -> bool:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_cpu_executor, verify_password_sync, plain_password, hashed_password)


USERS_DB = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "hashed_password": hash_password_sync("admin123"),
    },
    "demo": {
        "username": "demo",
        "full_name": "Demo User",
        "hashed_password": hash_password_sync("demo123"),
    },
}


class Token(BaseModel):
    access_token: str
    token_type: str
    username: str
    full_name: str


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401, detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in USERS_DB:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return USERS_DB[username]


# ---------------------------------------------------------------------------
# Staged Hybrid Filtering — Metadata Index
# ---------------------------------------------------------------------------

class ChunkMetadata:
    """Metadata for a single chunk, used for pre/post filtering."""
    __slots__ = ['chunk_id', 'page_num', 'section', 'word_count',
                 'tags', 'has_code', 'has_table', 'content_quality_score']

    def __init__(self, chunk_id: int, page_num: int, section: str,
                 word_count: int, tags: List[str], has_code: bool,
                 has_table: bool, content_quality_score: float):
        self.chunk_id = chunk_id
        self.page_num = page_num
        self.section = section
        self.word_count = word_count
        self.tags = tags
        self.has_code = has_code
        self.has_table = has_table
        self.content_quality_score = content_quality_score


class MetadataIndex:
    """Indexed metadata store for fast pre-filtering.

    Stage 1 (Pre-filter): page_range, section → O(1) lookup via inverted index
    Stage 3 (Post-filter): word_count, tags, quality → lightweight scan
    """

    def __init__(self):
        self.chunks: List[ChunkMetadata] = []
        # Inverted indexes for Stage 1 pre-filtering
        self.page_index: Dict[int, List[int]] = defaultdict(list)  # page_num → [chunk_ids]
        self.section_index: Dict[str, List[int]] = defaultdict(list)  # section → [chunk_ids]
        self.all_sections: List[str] = []
        self.total_pages: int = 0

    def add_chunk(self, meta: ChunkMetadata):
        self.chunks.append(meta)
        self.page_index[meta.page_num].append(meta.chunk_id)
        if meta.section:
            self.section_index[meta.section.lower()].append(meta.chunk_id)

    def finalize(self):
        """Called after all chunks are added. Builds summary info."""
        self.all_sections = sorted(set(
            c.section for c in self.chunks if c.section
        ))
        self.total_pages = max((c.page_num for c in self.chunks), default=0)

    def pre_filter(self, page_start: Optional[int] = None,
                   page_end: Optional[int] = None,
                   sections: Optional[List[str]] = None) -> Optional[set]:
        """Stage 1: Pre-filter using indexed metadata.
        Returns set of chunk_ids that pass the filter, or None if no filter applied.
        """
        candidate_ids = None

        # Filter by page range
        if page_start is not None or page_end is not None:
            page_ids = set()
            start = page_start or 1
            end = page_end or self.total_pages
            for page in range(start, end + 1):
                page_ids.update(self.page_index.get(page, []))
            candidate_ids = page_ids

        # Filter by section/department
        if sections:
            section_ids = set()
            for sec in sections:
                sec_lower = sec.lower().strip()
                for key, ids in self.section_index.items():
                    if sec_lower in key or key in sec_lower:
                        section_ids.update(ids)
            if candidate_ids is not None:
                candidate_ids = candidate_ids.intersection(section_ids)
            else:
                candidate_ids = section_ids

        return candidate_ids

    def post_filter(self, chunk_ids: List[int],
                    min_word_count: Optional[int] = None,
                    tags: Optional[List[str]] = None,
                    min_quality: Optional[float] = None) -> List[int]:
        """Stage 3: Post-filter on non-indexed attributes.
        Lightweight refinement on already-retrieved results.
        """
        filtered = []
        for cid in chunk_ids:
            if cid >= len(self.chunks):
                continue
            meta = self.chunks[cid]

            if min_word_count and meta.word_count < min_word_count:
                continue
            if tags:
                if not any(t.lower() in [tag.lower() for tag in meta.tags] for t in tags):
                    continue
            if min_quality and meta.content_quality_score < min_quality:
                continue

            filtered.append(cid)
        return filtered


# ---------------------------------------------------------------------------
# Document Store — Enhanced with metadata index
# ---------------------------------------------------------------------------

class DocumentStore:
    """Stores vector index + metadata index + raw chunks for a document."""

    def __init__(self, doc_id: str, vector_store, chunks: list,
                 metadata_index: MetadataIndex, doc_info: dict, bm25: BM25Okapi,
                 page_texts: Optional[Dict[int, str]] = None,
                 indexed_chunks: Optional[list] = None, marked: bool = False,
                 section_index: Optional["SectionIndex"] = None):
        self.doc_id = doc_id
        self.vector_store = vector_store
        # Raw LangChain Document objects; chunks[i].metadata["chunk_id"] == i
        self.chunks = chunks
        self.metadata_index = metadata_index
        self.doc_info = doc_info
        self.bm25 = bm25  # sparse index over the same chunks, same order
        self.page_texts = page_texts or {}  # 1-indexed page -> cleaned text, for sentence-level answers
        # chunks with their context line prepended: what is embedded, BM25-indexed and reranked
        self.indexed_chunks = indexed_chunks or chunks
        self.marked = marked  # headings are '## ' lines taken from the PDF's fonts
        self.section_index = section_index  # only for large documents (see TWO_STAGE_MIN_CHUNKS)


# ---------------------------------------------------------------------------
# Metadata Extraction Helpers
# ---------------------------------------------------------------------------

def _detect_section(text: str, page_num: int, marked: bool = False) -> str:
    """Detect section/topic from chunk content using heading patterns.
    `marked`: the document's headings are '## ' lines from its fonts, so trust only those."""
    lines = text.strip().split('\n')
    for line in lines[:3]:  # Check first 3 lines for headings
        line = line.strip()
        # Common heading patterns
        if re.match(r'^#{1,4}\s+', line):
            return re.sub(r'^#{1,4}\s+', '', line).strip()
        if marked:
            continue
        # Numbered heading ("2.1 Methods"), not a numbered list item that is a whole sentence
        if re.match(r'^\d+\.?\d*\.?\s+[A-Z]', line) and len(line) <= 80 and not line.endswith('.'):
            return re.sub(r'^\d+\.?\d*\.?\s+', '', line).strip()
        if line.isupper() and 3 < len(line) < 80:
            return line.title()
        if re.match(r'^[A-Z][a-z].*:$', line):
            return line.rstrip(':').strip()
        if len(lines) > 1 and _is_heading(line):
            return line.rstrip(':').strip()
    return f"Page {page_num}"


def _extract_tags(text: str) -> List[str]:
    """Extract topic tags from chunk content."""
    tags = []
    text_lower = text.lower()

    # Domain-specific tag detection
    tag_patterns = {
        "machine_learning": r'\b(machine learning|ml|neural network|deep learning|model training)\b',
        "statistics": r'\b(statistics|statistical|regression|correlation|hypothesis)\b',
        "data_processing": r'\b(data processing|etl|pipeline|preprocessing|cleaning)\b',
        "feature_engineering": r'\b(feature engineering|feature selection|feature extraction)\b',
        "evaluation": r'\b(evaluation|accuracy|precision|recall|f1.score|metrics)\b',
        "visualization": r'\b(visualization|plot|chart|graph|dashboard)\b',
        "nlp": r'\b(nlp|natural language|text mining|tokeniz|embedding)\b',
        "deployment": r'\b(deployment|production|api|docker|kubernetes)\b',
        "optimization": r'\b(optimization|hyperparameter|tuning|grid search)\b',
        "algorithm": r'\b(algorithm|decision tree|random forest|svm|xgboost|gradient)\b',
    }

    for tag, pattern in tag_patterns.items():
        if re.search(pattern, text_lower):
            tags.append(tag)

    return tags


def _compute_quality_score(text: str) -> float:
    """Compute content quality score (0-1) based on text characteristics."""
    score = 0.5  # Base score

    word_count = len(text.split())
    # Prefer chunks with substantial content
    if word_count > 50:
        score += 0.1
    if word_count > 100:
        score += 0.1

    # Penalize very short chunks
    if word_count < 20:
        score -= 0.3

    # Reward structured content (lists, numbered items)
    if re.search(r'^\s*[\-\•\*]\s', text, re.MULTILINE):
        score += 0.1
    if re.search(r'^\s*\d+[\.\)]\s', text, re.MULTILINE):
        score += 0.1

    # Penalize chunks that are mostly special characters or numbers
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.4:
        score -= 0.2

    # Reward complete sentences
    if text.strip().endswith(('.', '!', '?')):
        score += 0.05

    return max(0.0, min(1.0, score))


def _has_code_content(text: str) -> bool:
    """Detect if chunk contains code snippets."""
    indicators = ['def ', 'class ', 'import ', 'return ', 'function ',
                  'const ', 'var ', 'let ', '```', '>>>']
    return any(ind in text for ind in indicators)


def _has_table_content(text: str) -> bool:
    """Detect if chunk contains table-like content."""
    lines = text.split('\n')
    pipe_lines = sum(1 for l in lines if '|' in l and l.count('|') >= 2)
    tab_lines = sum(1 for l in lines if '\t' in l)
    return pipe_lines >= 2 or tab_lines >= 3


# ---------------------------------------------------------------------------
# Lazy Model Loading
# ---------------------------------------------------------------------------

class ModelManager:
    def __init__(self):
        self.embeddings = None
        self.cross_encoder = None
        self.qa_reader = None
        self.text_splitter = None
        self._loading = False
        self._loaded = False
        self._error = None
        self._lock = threading.Lock()

    @property
    def is_ready(self):
        return self._loaded and self._error is None

    @property
    def status(self):
        if self._loaded:
            return "ready"
        if self._error:
            return f"error: {self._error}"
        if self._loading:
            return "loading"
        return "not_started"

    def load_models(self):
        with self._lock:
            if self._loaded or self._loading:
                return
            self._loading = True

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_huggingface import HuggingFaceEmbeddings
            from sentence_transformers import CrossEncoder
            from transformers import pipeline as hf_pipeline

            logger.info("Loading embedding model: BAAI/bge-large-en-v1.5 ...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-large-en-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            logger.info("Loading cross-encoder reranker: BAAI/bge-reranker-base ...")
            self.cross_encoder = CrossEncoder("BAAI/bge-reranker-base")

            logger.info("Loading QA reader: deepset/deberta-v3-large-squad2 ...")
            self.qa_reader = hf_pipeline(
                "question-answering",
                model="deepset/deberta-v3-large-squad2",
                tokenizer="deepset/deberta-v3-large-squad2",
            )

            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=160,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
                length_function=len,
                add_start_index=True,  # chunk.metadata["start_index"]: offset in the page text
            )

            self._loaded = True
            self._loading = False
            logger.info("All models loaded successfully.")

        except Exception as e:
            self._error = str(e)
            self._loading = False
            logger.error(f"Model loading failed: {e}")
            import traceback
            traceback.print_exc()

    def require_models(self):
        if self._error:
            raise HTTPException(status_code=503,
                detail=f"Model loading failed: {self._error}. Check server logs.")
        if not self._loaded:
            raise HTTPException(status_code=503,
                detail="Models are still loading. Please wait and try again.")

    def rerank_documents(self, query: str, documents: list, top_n: int = 5) -> list:
        """Return [(doc, score)] best first, with score in 0-1 (usable as confidence)."""
        if not documents:
            return []
        import torch
        pairs = [[query, doc.page_content] for doc in documents]
        # Force a sigmoid so scores are 0-1 whatever the model's default activation is
        scores = self.cross_encoder.predict(pairs, activation_fct=torch.nn.Sigmoid())
        order = np.argsort(-np.asarray(scores, dtype=float))[:top_n]
        return [(documents[i], float(scores[i])) for i in order]


models = ModelManager()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    restored = _load_documents()
    users = _load_users()
    if restored or users:
        logger.info(f"Restored {restored} document(s) and {users} user(s) from {DATA_DIR}")
    logger.info("Server starting - models will load in background...")
    thread = threading.Thread(target=models.load_models, daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# In-Memory Stores
# ---------------------------------------------------------------------------

document_stores: Dict[str, DocumentStore] = {}
doc_metadata: dict = {}

# ---------------------------------------------------------------------------
# Vector backends, persistence and per-user ownership
# ---------------------------------------------------------------------------

DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
PERSIST = os.getenv("PERSIST", "1") == "1"                  # keep documents and users across restarts
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "qdrant")      # "qdrant" or "numpy" (exact, in-memory)
QDRANT_URL = os.getenv("QDRANT_URL", "")                    # empty: embedded on-disk Qdrant under DATA_DIR
QDRANT_COLLECTION = "smartpdf_chunks"
_QDRANT_NS = uuid.UUID("6f1c2f0e-8f55-4d2e-9d0b-0a7c3d3d9c11")
_qdrant = None
_DEFAULT_USERS = set(USERS_DB)
USERS_FILE_NAME = "users.json"


class NumpyVectors:
    """Exact cosine search over L2-normalised vectors held in memory (fallback when Qdrant is off)."""

    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype="float32")

    def search(self, query_vec, k: int, allowed_ids: Optional[set] = None) -> List[tuple]:
        scores = self.vectors @ np.asarray(query_vec, dtype="float32")
        if allowed_ids is not None:
            masked = np.full(len(scores), -np.inf, dtype="float32")
            idx = np.fromiter((i for i in allowed_ids if 0 <= i < len(scores)), dtype=int)
            masked[idx] = scores[idx]
            scores = masked
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if np.isfinite(scores[i])]


class QdrantVectors:
    """One document's chunk vectors in a shared Qdrant collection (payload: doc_id, owner, chunk_id).

    Exact filters on doc_id and chunk_id restrict the search to a document or a section scope, and
    the vectors survive restarts (embedded on-disk mode, or a Qdrant server via QDRANT_URL)."""

    def __init__(self, client, doc_id: str):
        self.client, self.doc_id = client, doc_id

    @classmethod
    def create(cls, client, doc_id: str, owner: str, vectors) -> "QdrantVectors":
        from qdrant_client import models as qm
        vectors = np.asarray(vectors, dtype="float32")
        if not client.collection_exists(QDRANT_COLLECTION):
            client.create_collection(QDRANT_COLLECTION, vectors_config=qm.VectorParams(
                size=int(vectors.shape[1]), distance=qm.Distance.COSINE))
            client.create_payload_index(QDRANT_COLLECTION, "doc_id", qm.PayloadSchemaType.KEYWORD)
            client.create_payload_index(QDRANT_COLLECTION, "chunk_id", qm.PayloadSchemaType.INTEGER)
        for start in range(0, len(vectors), 256):
            client.upsert(QDRANT_COLLECTION, points=[
                qm.PointStruct(id=str(uuid.uuid5(_QDRANT_NS, f"{doc_id}:{i}")), vector=vectors[i].tolist(),
                               payload={"doc_id": doc_id, "owner": owner, "chunk_id": int(i)})
                for i in range(start, min(start + 256, len(vectors)))])
        return cls(client, doc_id)

    def search(self, query_vec, k: int, allowed_ids: Optional[set] = None) -> List[tuple]:
        from qdrant_client import models as qm
        must = [qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=self.doc_id))]
        if allowed_ids is not None:
            must.append(qm.FieldCondition(key="chunk_id", match=qm.MatchAny(any=sorted(int(i) for i in allowed_ids))))
        res = self.client.query_points(QDRANT_COLLECTION, query=[float(x) for x in query_vec], limit=k,
                                       query_filter=qm.Filter(must=must), with_payload=["chunk_id"])
        return [(int(p.payload["chunk_id"]), float(p.score)) for p in res.points]


def _qdrant_client():
    global _qdrant
    if _qdrant is None:
        from qdrant_client import QdrantClient
        _qdrant = QdrantClient(url=QDRANT_URL) if QDRANT_URL else QdrantClient(path=os.path.join(DATA_DIR, "qdrant"))
    return _qdrant


def _close_qdrant():
    global _qdrant
    if _qdrant is not None:
        _qdrant.close()
        _qdrant = None


def _make_vector_backend(doc_id: str, owner: str, vectors):
    if VECTOR_BACKEND == "qdrant":
        try:
            return QdrantVectors.create(_qdrant_client(), doc_id, owner, vectors)
        except Exception as e:  # missing package, locked folder, server down...
            logger.warning(f"Qdrant unavailable ({e}); using in-memory exact search (not persisted)")
    return NumpyVectors(vectors)


def _doc_dir(doc_id: str) -> str:
    return os.path.join(DATA_DIR, "docs", doc_id)


def _write_json_atomic(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)


def _persist_document(doc_id: str, store: "DocumentStore", minors: List[str], vectors) -> None:
    """Everything except the vectors that live in Qdrant: enough to rebuild the store after a restart
    without re-embedding (BM25, the metadata index and the section index are rebuilt in seconds)."""
    if not PERSIST:
        return
    backend = "qdrant" if isinstance(store.vector_store, QdrantVectors) else "numpy"
    d = _doc_dir(doc_id)
    _write_json_atomic(os.path.join(d, "doc.json"), {
        "doc_id": doc_id, "doc_info": store.doc_info, "marked": store.marked, "backend": backend,
        "minors": minors,
        "chunks": [{"text": c.page_content, "metadata": c.metadata} for c in store.chunks],
        "indexed_texts": [c.page_content for c in store.indexed_chunks],
        "page_texts": {str(k): v for k, v in store.page_texts.items()},
        "sections": store.section_index.sections if store.section_index is not None else None,
    })
    if store.section_index is not None:
        np.save(os.path.join(d, "sections.npy"), store.section_index.vectors)
    if backend == "numpy":
        np.save(os.path.join(d, "vectors.npy"), np.asarray(vectors, dtype="float32"))


def _load_documents() -> int:
    """Rebuild every saved document at startup. Returns how many were restored."""
    from langchain_core.documents import Document
    root = os.path.join(DATA_DIR, "docs")
    if not PERSIST or not os.path.isdir(root):
        return 0
    loaded = 0
    for doc_id in sorted(os.listdir(root)):
        try:
            d = _doc_dir(doc_id)
            with open(os.path.join(d, "doc.json"), encoding="utf-8") as f:
                p = json.load(f)
            chunks = [Document(page_content=c["text"], metadata=c["metadata"]) for c in p["chunks"]]
            indexed = [Document(page_content=t, metadata=dict(c.metadata)) for c, t in zip(chunks, p["indexed_texts"])]
            vectors = (NumpyVectors(np.load(os.path.join(d, "vectors.npy"))) if p["backend"] == "numpy"
                       else QdrantVectors(_qdrant_client(), doc_id))
            section_index = (SectionIndex(p["sections"], np.load(os.path.join(d, "sections.npy")))
                             if p.get("sections") else None)
            document_stores[doc_id] = DocumentStore(
                doc_id=doc_id, vector_store=vectors, chunks=chunks,
                metadata_index=_build_metadata_index(chunks, p["minors"], p["marked"]),
                doc_info=p["doc_info"], bm25=_build_bm25(indexed),
                page_texts={int(k): v for k, v in p["page_texts"].items()},
                indexed_chunks=indexed, marked=p["marked"], section_index=section_index)
            doc_metadata[doc_id] = p["doc_info"]
            loaded += 1
        except Exception as e:
            logger.warning(f"Could not restore document {doc_id}: {e}")
    return loaded


def _save_users() -> None:
    if PERSIST:
        _write_json_atomic(os.path.join(DATA_DIR, USERS_FILE_NAME),
                           {u: v for u, v in USERS_DB.items() if u not in _DEFAULT_USERS})


def _load_users() -> int:
    path = os.path.join(DATA_DIR, USERS_FILE_NAME)
    if not PERSIST or not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    USERS_DB.update(saved)
    return len(saved)


def _owned_store(doc_id: str, user: dict) -> "DocumentStore":
    """The user's own document. Someone else's document looks exactly like a missing one."""
    store = document_stores.get(doc_id)
    if store is None or store.doc_info.get("uploaded_by") != user["username"]:
        raise HTTPException(status_code=404, detail="Document not found. Please upload a PDF first.")
    return store


def _build_metadata_index(chunks: list, minors: List[str], marked: bool) -> "MetadataIndex":
    metadata_index = MetadataIndex()
    for i, chunk in enumerate(chunks):
        page_num = chunk.metadata.get("page", 0) + 1  # 1-indexed
        text = chunk.page_content
        section = _detect_section(text, page_num, marked)
        if section == f"Page {page_num}" and minors[i]:
            section = minors[i]  # no heading inside the chunk: use the heading above it
        metadata_index.add_chunk(ChunkMetadata(
            chunk_id=i, page_num=page_num, section=section, word_count=len(text.split()),
            tags=_extract_tags(text), has_code=_has_code_content(text),
            has_table=_has_table_content(text), content_quality_score=_compute_quality_score(text)))
    metadata_index.finalize()
    return metadata_index


perf_metrics = {
    "total_queries": 0,
    "total_uploads": 0,
    "total_chunks_indexed": 0,
    "avg_retrieval_time_ms": 0.0,
    "avg_embedding_time_ms": 0.0,
    "query_times": [],
    "embedding_times": [],
    "prefilter_times": [],
    "postfilter_times": [],
}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    doc_id: str
    question: str
    # Stage 1: Pre-filter options (indexed, fast)
    page_start: Optional[int] = Field(None, description="Filter: start page")
    page_end: Optional[int] = Field(None, description="Filter: end page")
    sections: Optional[List[str]] = Field(None, description="Filter: sections/topics")
    # Stage 3: Post-filter options (non-indexed, lightweight)
    min_word_count: Optional[int] = Field(None, description="Post-filter: minimum words per chunk")
    tags: Optional[List[str]] = Field(None, description="Post-filter: topic tags")
    min_quality: Optional[float] = Field(None, description="Post-filter: min quality score 0-1")


class RegisterRequest(BaseModel):
    username: str
    password: str
    full_name: str


# ---------------------------------------------------------------------------
# POST /login
# ---------------------------------------------------------------------------

@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = USERS_DB.get(form_data.username)
    if not user or not await verify_password_async(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": form_data.username})
    return {
        "access_token": access_token, "token_type": "bearer",
        "username": user["username"], "full_name": user["full_name"],
    }


# ---------------------------------------------------------------------------
# POST /register
# ---------------------------------------------------------------------------

@app.post("/register")
async def register(req: RegisterRequest):
    if req.username in USERS_DB:
        raise HTTPException(status_code=400, detail="Username already exists")
    if len(req.password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters")
    hashed = await hash_password_async(req.password)
    USERS_DB[req.username] = {
        "username": req.username, "full_name": req.full_name, "hashed_password": hashed,
    }
    _save_users()
    access_token = create_access_token(data={"sub": req.username})
    return {
        "access_token": access_token, "token_type": "bearer",
        "username": req.username, "full_name": req.full_name,
    }


# ---------------------------------------------------------------------------
# POST /upload/ — Upload and index a PDF with metadata extraction
# ---------------------------------------------------------------------------

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    models.require_models()

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)}MB.")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _cpu_executor, _process_pdf_sync, content, file.filename, user["username"],
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


def _process_pdf_sync(content: bytes, filename: str, username: str) -> dict:
    """Process PDF: extract text, build chunks, create metadata index + vector store."""
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_core.documents import Document

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            pages = _extract_pages(tmp_path)  # reading order, tables rebuilt as labelled rows
        except Exception as e:
            logger.warning(f"Table-aware extraction failed ({e}); falling back to plain loader")
            pages = PyMuPDFLoader(tmp_path).load()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not pages or all(p.page_content.strip() == "" for p in pages):
        raise HTTPException(status_code=400,
            detail="Could not extract text from PDF. The file may be scanned/image-based.")

    for page in pages:
        page.page_content = _clean_text(page.page_content)
    page_texts = {p.metadata.get("page", 0) + 1: p.page_content for p in pages}
    marked = sum(ln.startswith("## ") for t in page_texts.values() for ln in t.split(chr(10))) >= 3

    # Split into chunks
    chunks = models.text_splitter.split_documents(pages)
    if len(chunks) == 0:
        raise HTTPException(status_code=400, detail="PDF produced no usable text chunks")

    # Stable chunk IDs, stored in each chunk's metadata so search hits map straight
    # back to chunks[i] and the metadata index (no text matching).
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    # Context line per chunk ("Title > Part > nearest heading"): indexed together with the text so
    # a chunk of bare spec lines still says what it is; the displayed chunk stays unchanged.
    title = _doc_title(page_texts, filename)
    contexts = _chunk_contexts(chunks, page_texts, title, marked)
    indexed_chunks = [
        Document(page_content=f"{ctx}\n{c.page_content}" if ctx else c.page_content, metadata=dict(c.metadata))
        for c, (ctx, _heading) in zip(chunks, contexts)
    ]

    minors = [heading for _ctx, heading in contexts]
    metadata_index = _build_metadata_index(chunks, minors, marked)

    # Embed every chunk (with its context line) and hand the vectors to the vector backend
    t_embed_start = time.time()
    doc_id = str(uuid.uuid4())
    vectors = np.asarray(models.embeddings.embed_documents([c.page_content for c in indexed_chunks]), dtype="float32")
    vector_store = _make_vector_backend(doc_id, username, vectors)
    bm25 = _build_bm25(indexed_chunks)
    section_index = (_build_section_index(chunks, page_texts, title, marked)
                     if TWO_STAGE and len(chunks) >= TWO_STAGE_MIN_CHUNKS else None)
    t_embed_end = time.time()
    embedding_time_ms = (t_embed_end - t_embed_start) * 1000

    # Store document
    doc_info = {
        "filename": filename,
        "num_chunks": len(chunks),
        "num_pages": len(pages),
        "uploaded_by": username,
        "sections": metadata_index.all_sections,
        "embedding_time_ms": round(embedding_time_ms, 2),
    }
    store = document_stores[doc_id] = DocumentStore(
        doc_id=doc_id, vector_store=vector_store,
        chunks=chunks, metadata_index=metadata_index, doc_info=doc_info, bm25=bm25,
        page_texts=page_texts, indexed_chunks=indexed_chunks, marked=marked,
        section_index=section_index,
    )
    doc_metadata[doc_id] = doc_info
    try:
        _persist_document(doc_id, store, minors, vectors)
    except Exception as e:
        logger.warning(f"Could not save document {doc_id} to disk: {e}")

    # Update metrics
    perf_metrics["total_uploads"] += 1
    perf_metrics["total_chunks_indexed"] += len(chunks)
    perf_metrics["embedding_times"].append(embedding_time_ms)
    perf_metrics["avg_embedding_time_ms"] = round(
        sum(perf_metrics["embedding_times"]) / len(perf_metrics["embedding_times"]), 2)

    logger.info(
        f"Indexed {len(chunks)} chunks from '{filename}' in {embedding_time_ms:.0f}ms "
        f"(sections={len(metadata_index.all_sections)}, doc_id={doc_id})")

    return {
        "doc_id": doc_id,
        "filename": filename,
        "num_chunks": len(chunks),
        "num_pages": len(pages),
        "sections": metadata_index.all_sections[:20],
        "embedding_time_ms": round(embedding_time_ms, 2),
        "message": (
            f"PDF processed successfully. {len(chunks)} chunks indexed. "
            f"{len(metadata_index.all_sections)} sections detected."
        ),
    }


# ---------------------------------------------------------------------------
# POST /ask/ — 3-Stage Hybrid Filtering Pipeline
# ---------------------------------------------------------------------------

@app.post("/ask/")
async def ask_question(req: AskRequest, user: dict = Depends(get_current_user)):
    models.require_models()

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    _owned_store(req.doc_id, user)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _cpu_executor, _ask_question_sync,
            req.doc_id, req.question, user["username"],
            req.page_start, req.page_end, req.sections,
            req.min_word_count, req.tags, req.min_quality,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ask error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


def _is_broad_question(question: str) -> bool:
    """Detect broad/descriptive questions vs factoid questions."""
    broad_indicators = [
        "tell me about", "explain", "describe", "what are the",
        "list the", "summarize", "overview", "different types",
        "how does", "what is the process", "walk me through",
        "elaborate", "discuss", "compare", "contrast",
        "advantages", "disadvantages", "pros and cons",
        "techniques", "methods", "approaches", "strategies",
        "steps", "stages", "phases", "components",
        "what all", "give me", "provide", "mention",
        "types of", "kinds of", "examples of",
    ]
    q_lower = question.lower().strip()
    return any(indicator in q_lower for indicator in broad_indicators)


# ---------------------------------------------------------------------------
# Structured extractive answers (no LLM): whole sentences, document order,
# grouped by section/page with page numbers
# ---------------------------------------------------------------------------

MAX_ANSWER_ITEMS = 16      # bullets in one structured answer
MAX_ITEMS_PER_CHUNK = 4    # so every top chunk is represented, not just the first
MAX_ANSWER_ITEMS_BROAD = 32    # "summarize / explain / list" questions want breadth
MAX_ITEMS_PER_CHUNK_BROAD = 12
# A chunk only contributes if its reranker score is at least this fraction of the best chunk's
ANSWER_MIN_RELATIVE_SCORE = 0.25
_SENT_RE = re.compile(r".+?(?:[.!?][\"')\]]*(?=\s+[A-Z0-9•\"(\[])|$)")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[•\-\*]|\d+[.)])\s+\S")
_ABBREV_END_RE = re.compile(r"\b(vs|e\.g|i\.e|etc|fig|figs|eq|cf|dr|mr|mrs|approx)\.$", re.I)
# Symbol-font / private-use bullets that PDFs extract instead of "•"
_BULLET_GLYPHS = str.maketrans({c: "•" for c in "●▪◦‣⁃"})
_QUESTION_NOISE = frozenset(
    "a an and are as at be by for from has have how in is it of on or that the this to was what when "
    "where which who why will with about tell me explain describe summarize summarise give provide list "
    "mention overview different types type kinds examples elaborate discuss can could does do you your".split())


_TOP_NUMBERED_RE = re.compile(r"^\d+\s+[A-Z]")   # "3 Fine-tuning": a top-level numbered section
_HEADING_RE = re.compile(r"^(part|chapter|section|appendix|module|unit)\s+[\dIVXivx]+\b", re.I)


def _is_heading(text: str, marked: bool = False) -> bool:
    """A heading line. "## " marks typographic headings found from the PDF's fonts; when the
    document has them (`marked`) nothing else counts. Otherwise a text heuristic is used:
    short, unpunctuated, title-like ("Part 4: ...", "What Exists in Research")."""
    s = text.strip()
    if s.startswith("## "):
        return len(s) > 3
    if marked:
        return False
    words = s.split()
    if not s or len(s) > 80 or len(words) > 12 or " · " in s:
        return False
    if s.startswith(("•", "-", "*")) or not (s[0].isupper() or s[0].isdigit()):
        return False
    if s.endswith((".", "!", ",", ";")):
        return False
    alpha = [w for w in words if w[0].isalpha()]
    title_like = bool(alpha) and sum(w[0].isupper() for w in alpha) / len(alpha) >= 0.6
    if s.endswith("?"):  # question-style headings must look like titles; short questions are prose
        return bool(_HEADING_RE.match(s)) or title_like
    return bool(_HEADING_RE.match(s)) or s.endswith(":") or title_like or len(words) <= 4


def _clean_text(text: str) -> str:
    """Repair common PDF-extraction damage before chunking and embedding."""
    text = unicodedata.normalize("NFKC", text)             # ﬀ -> ff, ﬁ -> fi, nbsp -> space
    text = text.translate(_BULLET_GLYPHS)                  # symbol-font bullets -> "•"
    # glyphs the PDF font could not map: U+FFFD, plus the noncharacters U+FFFE/U+FFFF/U+FDD0-FDEF
    text = re.sub("[�￾￿﷐-﷯\x00]", "", text)
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)        # "trans-\nformers" -> "transformers"
    text = re.sub(r"(?<![.!?:;])\n(?=[a-z])", " ", text)   # re-join hard-wrapped lines
    return re.sub(r"[ \t]+", " ", text)


def _segment_page(text: str, marked: bool = False) -> List[tuple]:
    """Split text into (start, end, segment) units: headings, list items/table rows, sentences."""
    lines = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[^\n]+", text) if m.group().strip()]
    segments: List[tuple] = []
    for start, end, line in lines:
        # Headings, list items and short lines are self-contained: splitting them leaves fragments
        if _is_heading(line, marked) or len(line.strip()) <= 100 or _LIST_ITEM_RE.match(line):
            segments.append((start, end, line.strip()))
            continue
        parts: List[list] = []
        for m in _SENT_RE.finditer(line):
            seg = m.group().strip()
            if not seg:
                continue
            if parts and _ABBREV_END_RE.search(parts[-1][2]):  # "vs." / "e.g." is not a sentence end
                parts[-1][1] = start + m.end()
                parts[-1][2] += " " + seg
            else:
                parts.append([start + m.start(), start + m.end(), seg])
        segments.extend(tuple(p) for p in parts)
    return segments


def _heading_above(segments: list, k: int, marked: bool = False) -> Optional[str]:
    """Nearest heading-like segment before segments[k] on the same page, if any."""
    for j in range(k - 1, -1, -1):
        if _is_heading(segments[j][2], marked):
            return _heading_label(segments[j][2])
    return None


# ---------------------------------------------------------------------------
# Table-aware page extraction and per-chunk context
# ---------------------------------------------------------------------------

_ROW_TOL = 3.0             # pt: lines whose vertical centres differ by less share a visual row
_ROW_GAP = 16.0            # pt: max vertical gap between consecutive lines of one table
_MAX_CELL_CHARS = 60       # longer lines are prose, not table cells (also keeps 2-column text out)
_MAX_AVG_CELL_CHARS = 40
_MAX_HEADER_CHARS = 40


def _detect_tables(lines: list) -> list:
    """Find borderless tables from text-line geometry.

    lines: [(x0, y0, x1, y1, text)] in reading order. A table is a header row of 2+ short cells
    followed by 2+ rows whose cells sit under the header columns. A line under a column with no
    first-column cell continues the cell above it (wrapped text).
    Returns [{"members": {line index}, "rows": ["• Header: cell; Header: cell", ...]}].
    """
    n = len(lines)
    if n < 6:
        return []
    cy = [(l[1] + l[3]) / 2 for l in lines]
    visual: List[List[int]] = []
    for i in sorted(range(n), key=lambda i: (cy[i], lines[i][0])):
        if visual and abs(cy[i] - cy[visual[-1][0]]) <= _ROW_TOL:
            visual[-1].append(i)
        else:
            visual.append([i])
    for row in visual:
        row.sort(key=lambda i: lines[i][0])

    center = lambda i: (lines[i][0] + lines[i][2]) / 2
    short = lambda row: all(len(lines[i][4]) <= _MAX_CELL_CHARS for i in row)

    tables: List[dict] = []
    r = 0
    while r < len(visual):
        header = visual[r]
        if len(header) < 2 or not short(header) or any(len(lines[i][4]) > _MAX_HEADER_CHARS for i in header):
            r += 1
            continue
        # A real header is made of labels. If half its cells are numbers ("184320", "1.0T", "7B")
        # this is a data row (the true header is split over several lines): don't guess labels.
        if 2 * sum(sum(ch.isdigit() for ch in lines[i][4]) >= 0.4 * len(lines[i][4]) for i in header) >= len(header):
            r += 1
            continue
        anchors = [center(i) for i in header]
        column = lambda i: min(range(len(anchors)), key=lambda c: abs(anchors[c] - center(i)))
        data: List[List[str]] = []
        members = set(header)
        bottom = max(lines[i][3] for i in header)
        end = r + 1
        while end < len(visual):
            row = visual[end]
            if min(lines[i][1] for i in row) - bottom > _ROW_GAP or not short(row):
                break
            cols = [column(i) for i in row]
            if len(set(cols)) != len(cols):
                break
            if 0 in cols:                          # first-column cell: a new data row
                cells = [""] * len(anchors)
                for i, c in zip(row, cols):
                    cells[c] = lines[i][4]
                data.append(cells)
            elif data:                             # wrapped text of the row above
                for i, c in zip(row, cols):
                    data[-1][c] = f"{data[-1][c]} {lines[i][4]}".strip()
            else:
                break
            members.update(row)
            bottom = max(lines[i][3] for i in row)
            end += 1

        header_text = [lines[i][4] for i in header]
        cells_all = header_text + [c for row in data for c in row if c]
        wide_rows = sum(sum(bool(c) for c in row) >= 2 for row in data)
        if (len(data) >= 2 and wide_rows >= 2
                and sum(len(c) for c in cells_all) / len(cells_all) <= _MAX_AVG_CELL_CHARS):
            rows = ["• " + "; ".join(f"{header_text[c]}: {cell}" for c, cell in enumerate(row) if cell)
                    for row in data]
            rendered = " ".join(rows)
            # Never lose information: if any data cell's text is missing from the rebuilt rows
            # (merged cells, multi-row headers), keep the original lines instead.
            if all(lines[i][4] in rendered for i in members - set(header)):
                tables.append({"members": members, "rows": rows})
                r = end
            else:
                r += 1
        else:
            r += 1
    return tables


def _margin_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()  # "Page 3" and "Page 4" are the same footer


def _page_lines(page) -> List[tuple]:
    """(x0, y0, x1, y1, text, font_size, bold) for each text line, in the PDF's reading order."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not text or not spans:
                continue
            x0, y0, x1, y1 = line["bbox"]
            size = round(max(spans, key=lambda s: len(s["text"]))["size"] * 2) / 2
            bold = all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans)
            lines.append((x0, y0, x1, y1, text, size, bold))
    return lines


def _repeated_margin_lines(pdf, all_lines: Optional[list] = None) -> set:
    """Header/footer text: lines in the top or bottom 10% of a page that repeat on 3+ pages
    (and at least half of them). Left in, they become fake headings and noise in every chunk."""
    found: Dict[str, set] = defaultdict(set)
    for number, page in enumerate(pdf):
        height = page.rect.height
        for line in (all_lines[number] if all_lines is not None else _page_lines(page)):
            if line[3] < 0.10 * height or line[1] > 0.90 * height:
                found[_margin_key(line[4])].add(number)
    need = max(3, (len(pdf) + 1) // 2)
    return {key for key, pages in found.items() if len(pages) >= need}


def _body_font_size(all_lines: list) -> float:
    """The document's body text size: the most common font size, weighted by characters."""
    weight: Dict[float, int] = defaultdict(int)
    for lines in all_lines:
        for line in lines:
            weight[line[5]] += len(line[4])
    return max(weight, key=weight.get) if weight else 0.0


def _font_heading(line: tuple, body: float) -> bool:
    """A heading by typography: clearly larger than body text, or a short bold line."""
    text, size, bold = line[4].strip(), line[5], line[6]
    if not (3 <= len(text) <= 120) or text.endswith((".", ",", ";")):
        return False
    if text.startswith(("•", "-", "*", "–")) or sum(c.isalpha() for c in text) < 0.5 * len(text):
        return False  # list item, or mostly digits/symbols (page number, equation, table cell)
    if size >= body + 1.0:
        return True
    return bold and size >= body - 0.5 and len(text) <= 80 and (text[0].isupper() or text[0].isdigit())


def _page_text_with_tables(page, skip: frozenset = frozenset(), body_size: float = 0.0,
                           lines: Optional[list] = None) -> str:
    """Page text in reading order, with each detected table rewritten as one
    self-describing row per line ("Domain: DSA; Data: Excellent; ...") and, when `body_size`
    is given, each typographic heading prefixed "## ". Lines whose _margin_key is in `skip`
    (repeated headers/footers) are dropped."""
    source = lines if lines is not None else _page_lines(page)
    height = page.rect.height
    # Only text that really sits in the margin is a header/footer: a bare number in a table cell
    # has the same key as a page number, but is body content.
    lines = [l for l in source
             if not (skip and _margin_key(l[4]) in skip and (l[3] < 0.10 * height or l[1] > 0.90 * height))]
    tables = _detect_tables(lines)
    owner = {i: t for t, table in enumerate(tables) for i in table["members"]}
    out: List[str] = []
    emitted: set = set()
    for i, line in enumerate(lines):
        t = owner.get(i)
        if t is None:
            out.append("## " + line[4] if body_size and _font_heading(line, body_size) else line[4])
        elif t not in emitted:
            emitted.add(t)
            out.extend(tables[t]["rows"])
    return "\n".join(out)


def _extract_pages(path: str) -> list:
    """One LangChain Document per page (metadata.page is 0-indexed): tables rebuilt, repeated
    headers/footers dropped, typographic headings marked "## " (only if the document has 3+)."""
    import pymupdf
    from langchain_core.documents import Document

    with pymupdf.open(path) as pdf:
        all_lines = [_page_lines(page) for page in pdf]
        skip = frozenset(_repeated_margin_lines(pdf, all_lines))
        body = _body_font_size([[l for l in ls if _margin_key(l[4]) not in skip] for ls in all_lines])
        texts = [_page_text_with_tables(page, skip, body, ls) for page, ls in zip(pdf, all_lines)]
        total = len(pdf)
    if sum(ln.startswith("## ") for t in texts for ln in t.split("\n")) < 3:
        texts = [re.sub(r"(?m)^## ", "", t) for t in texts]   # too few to trust: fall back to text rules
    return [Document(page_content=t, metadata={"source": path, "page": i, "total_pages": total})
            for i, t in enumerate(texts)]


def _heading_label(text: str) -> str:
    """Heading text without the '## ' marker or a trailing colon."""
    return re.sub(r"^##\s+", "", text.strip()).rstrip(":").strip()


def _visible(text: str) -> str:
    """Chunk text as users should see it (no heading markers)."""
    return re.sub(r"(?m)^## ", "", text)


def _doc_title(page_texts: Dict[int, str], filename: str) -> str:
    first = next((_heading_label(ln) for ln in (page_texts.get(1) or "").split("\n") if ln.strip()), "")
    if first and len(first) <= 120 and not first.endswith("."):
        return first
    return re.sub(r"[_\-]+", " ", os.path.splitext(filename)[0]).strip()


def _document_headings(page_texts: Dict[int, str], marked: bool = False) -> List[tuple]:
    """(page, offset, label, is_major) for every heading, in document order."""
    flat: List[tuple] = []
    for page in sorted(page_texts):
        for start, _end, text in _segment_page(page_texts[page], marked):
            if _is_heading(text, marked):
                label = _heading_label(text)
                flat.append((page, start, label, bool(_HEADING_RE.match(label) or _TOP_NUMBERED_RE.match(label))))
    return flat


class SectionIndex:
    """The sections of a large document (heading to heading, at most SECTION_MAX_CHUNKS chunks
    each). They are searched first, so the chunk search only runs where the answer can be."""

    def __init__(self, sections: List[dict], vectors):
        self.sections = sections          # [{"title": str, "chunk_ids": [int], "text": str}]
        self.vectors = vectors            # (n, d) float32, L2-normalised
        self.bm25 = BM25Okapi([_tokenize(f"{s['title']} {s['text']}") or ["∅"] for s in sections])


def _split_into_sections(chunks: list, page_texts: Dict[int, str], title: str, marked: bool = False) -> List[dict]:
    flat = _document_headings(page_texts, marked)
    keys = [(f[0], f[1]) for f in flat]
    buckets: Dict[int, List[int]] = defaultdict(list)
    for chunk in chunks:
        page = chunk.metadata.get("page", 0) + 1
        pos = bisect.bisect_right(keys, (page, chunk.metadata.get("start_index", 0) or 0))
        buckets[pos].append(chunk.metadata["chunk_id"])
    sections: List[dict] = []
    for pos in sorted(buckets):
        label = flat[pos - 1][2] if pos else title
        major = next((flat[j][2] for j in range(pos - 1, -1, -1) if flat[j][3]), "")
        path = " > ".join(dict.fromkeys(p for p in (major, label) if p))
        ids = buckets[pos]
        for j in range(0, len(ids), SECTION_MAX_CHUNKS):   # keeps every section focused
            part = ids[j:j + SECTION_MAX_CHUNKS]
            sections.append({"title": path, "chunk_ids": part,
                             "text": " ".join(_visible(chunks[i].page_content) for i in part)})
    return sections


def _build_section_index(chunks: list, page_texts: Dict[int, str], title: str, marked: bool = False) -> SectionIndex:
    sections = _split_into_sections(chunks, page_texts, title, marked)
    texts = [f"{s['title']}\n{s['text'][:SECTION_EMBED_CHARS]}" for s in sections]
    return SectionIndex(sections, np.asarray(models.embeddings.embed_documents(texts), dtype="float32"))


def _section_scope(index: SectionIndex, question: str, query_vec) -> set:
    """Chunk ids of the TOP_SECTIONS sections that best match the question (dense + BM25, RRF)."""
    depth = TOP_SECTIONS * 2
    dense = np.argsort(-(index.vectors @ np.asarray(query_vec, dtype="float32")))[:depth].tolist()
    tokens = _tokenize(question)
    scores = index.bm25.get_scores(tokens) if tokens else np.zeros(len(index.sections))
    sparse = [int(i) for i in np.argsort(-scores)[:depth] if scores[i] > 0]
    top = _rrf_fuse([dense, sparse])[:TOP_SECTIONS]
    return {cid for s in top for cid in index.sections[s]["chunk_ids"]}


def _chunk_contexts(chunks: list, page_texts: Dict[int, str], title: str,
                    marked: bool = False) -> List[tuple]:
    """(context, nearest_heading) per chunk.

    The context line ("Title > Part 5: ... > Hyperparameters") is prepended to the chunk text
    for embedding, BM25 and reranking, so a chunk of bare spec lines still says what it is.
    Chunks need metadata["start_index"] (set by the text splitter).
    """
    flat = _document_headings(page_texts, marked)  # (page, offset, label, is_major)
    keys = [(f[0], f[1]) for f in flat]

    out: List[tuple] = []
    for chunk in chunks:
        page = chunk.metadata.get("page", 0) + 1
        pos = bisect.bisect_right(keys, (page, chunk.metadata.get("start_index", 0) or 0))
        minor = flat[pos - 1][2] if pos else ""
        major = next((flat[j][2] for j in range(pos - 1, -1, -1) if flat[j][3]), "")
        parts: List[str] = []
        for part in (title, major, minor):
            if part and part not in parts and part not in chunk.page_content:  # nothing repeated
                parts.append(part)
        out.append((" > ".join(parts), minor))
    return out


def _candidate_pool(dense_ids: List[int], sparse_ids: List[int]) -> List[int]:
    """Candidates for the reranker: the RRF top, plus each retriever's own best.

    Fusing then cutting at FUSED_TOP_K lets one weak ranker push a good match out: a chunk BM25
    ranks 12th but dense ranks 190th falls below the cut. So each retriever is also guaranteed
    its top GUARANTEED_PER_RETRIEVER (BM25 first: it is precise on numbers and exact terms).
    """
    pool = _rrf_fuse([dense_ids, sparse_ids])[:FUSED_TOP_K]
    seen = set(pool)
    for ids in (sparse_ids, dense_ids):
        for chunk_id in ids[:GUARANTEED_PER_RETRIEVER]:
            if len(pool) >= CANDIDATE_CAP:
                return pool
            if chunk_id not in seen:
                seen.add(chunk_id)
                pool.append(chunk_id)
    return pool


def _stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def _build_structured_answer(question: str, ranked: list, doc_store: "DocumentStore",
                             broad: bool = False) -> dict:
    """Turn reranked chunks into whole-sentence bullets grouped by section and page.

    Each chunk is widened to full sentences using its start offset in the page text
    (chunks cut mid-sentence), overlapping chunks claim each sentence only once, and the
    result is ordered as in the document rather than by relevance.
    """
    marked = getattr(doc_store, "marked", False)
    max_items = MAX_ANSWER_ITEMS_BROAD if broad else MAX_ANSWER_ITEMS
    per_chunk = MAX_ITEMS_PER_CHUNK_BROAD if broad else MAX_ITEMS_PER_CHUNK
    terms = {_stem(t) for t in _tokenize(question) if len(t) > 2 and t not in _QUESTION_NOISE}
    page_segments: Dict[int, list] = {}
    claimed: set = set()
    seen_texts: set = set()
    picked: List[tuple] = []  # (page, position, section, text)

    # Weak chunks only add noise to an answer: keep those close to the best one
    if ranked:
        floor = ANSWER_MIN_RELATIVE_SCORE * ranked[0][1]  # the top chunk already passed MIN_RERANK_SCORE
        ranked = [(d, s) for d, s in ranked if s >= floor]

    for doc, _score in ranked:
        if len(picked) >= max_items:
            break
        chunk_id = doc.metadata.get("chunk_id")
        page = doc.metadata.get("page", 0) + 1
        section = (doc_store.metadata_index.chunks[chunk_id].section
                   if chunk_id is not None else f"Page {page}")
        page_text = (doc_store.page_texts or {}).get(page)
        start = doc.metadata.get("start_index", -1)

        if page_text is not None and start is not None and start >= 0:
            segments = page_segments.setdefault(page, _segment_page(page_text, marked))
            lo, hi, key = start, start + len(doc.page_content), page
        else:  # no offsets: fall back to the chunk's own text
            segments = _segment_page(doc.page_content, marked)
            lo, hi, key = 0, len(doc.page_content), ("chunk", chunk_id)

        candidates = []
        for k, (s_start, s_end, text) in enumerate(segments):
            if s_end <= lo or s_start >= hi or (key, k) in claimed:
                continue
            text = re.sub(r"^[•\-\*]\s*", "", text)  # the UI draws its own bullets
            if (len(text) < 20 or text == section or _is_heading(text, marked) or text.lower() in seen_texts
                    or sum(c.isalpha() for c in text) < 0.4 * len(text)):
                continue
            seg_terms = {_stem(t) for t in _tokenize(text)}
            overlap = len(terms & seg_terms) / len(terms) if terms else 0.0
            candidates.append((overlap, k, s_start, text, _heading_above(segments, k, marked) or section))

        chosen = sorted(candidates, key=lambda c: (-c[0], c[1]))[:per_chunk]
        if not broad and chosen and chosen[0][0] == 0:
            chosen = chosen[:2]  # nothing matches the question's words: keep it short
        for _overlap, k, s_start, text, heading in chosen:
            if text.lower() in seen_texts:  # identical sentence already picked (picked in this loop)
                continue
            claimed.add((key, k))
            seen_texts.add(text.lower())
            picked.append((page, s_start, heading, text))

    if not picked:
        return {"answer": "No relevant content found.", "answer_blocks": [], "method": "none"}

    blocks: List[dict] = []
    by_key: Dict[tuple, dict] = {}
    for page, _pos, section, text in sorted(picked):
        block = by_key.get((section, page))
        if block is None:
            block = by_key[(section, page)] = {"heading": section, "page": page, "items": []}
            blocks.append(block)
        block["items"].append(text)

    lines: List[str] = []
    for b in blocks:
        generic = b["heading"] == f"Page {b['page']}"
        lines.append(b["heading"] if generic else f"{b['heading']} (p. {b['page']})")
        lines.extend(f"• {item}" for item in b["items"])
        lines.append("")
    return {"answer": "\n".join(lines).strip(), "answer_blocks": blocks, "method": "semantic_retrieval"}


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _build_bm25(chunks: list) -> BM25Okapi:
    # "∅" placeholder keeps BM25Okapi from dividing by zero on all-punctuation chunks
    return BM25Okapi([_tokenize(c.page_content) or ["∅"] for c in chunks])


def _dense_search(doc_store: DocumentStore, question: str,
                  allowed_ids: Optional[set], k: int, query_vec=None) -> List[int]:
    """Vector search -> chunk_ids, best first. With a pre-filter or section scope, only chunks in
    `allowed_ids` are searched, so the top-k always comes from the allowed subset."""
    if query_vec is None:
        query_vec = models.embeddings.embed_query(question)
    return [chunk_id for chunk_id, _score in doc_store.vector_store.search(query_vec, k, allowed_ids)]


def _bm25_search(doc_store: DocumentStore, question: str,
                 allowed_ids: Optional[set], k: int) -> List[int]:
    """BM25 keyword search -> chunk_ids, best first. Chunks with no term match are dropped."""
    query_tokens = _tokenize(question)
    if not query_tokens:
        return []
    scores = doc_store.bm25.get_scores(query_tokens)
    hits: List[int] = []
    for i in np.argsort(-scores):
        if scores[i] <= 0:
            break
        if allowed_ids is not None and int(i) not in allowed_ids:
            continue
        hits.append(int(i))
        if len(hits) >= k:
            break
    return hits


def _rrf_fuse(rankings: List[List[int]], k: int = RRF_K) -> List[int]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d))."""
    fused: Dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] += 1.0 / (k + rank)
    return sorted(fused, key=lambda cid: (-fused[cid], cid))


def _empty_result(message: str, started: float, stages: dict,
                  method: str = "none", score: float = 0.0,
                  source_chunks: Optional[List[str]] = None,
                  answer_blocks: Optional[List[dict]] = None) -> dict:
    """Uniform 'nothing to answer with' response (keeps the UI's stage panel intact)."""
    elapsed = round((time.time() - started) * 1000, 2)
    return {
        "answer": message,
        "answer_blocks": answer_blocks or None,  # closest sections, shown as "related" when not found
        "score": round(float(score), 4),
        "method": method,
        "answer_found": False,
        "source_chunks": source_chunks or [],
        "retrieval_pipeline": RETRIEVAL_PIPELINE_LABEL,
        "pipeline_stages": stages,
        "retrieval_time_ms": elapsed,
        "total_time_ms": elapsed,
    }


def _ask_question_sync(doc_id: str, question: str, username: str,
                       page_start, page_end, sections,
                       min_word_count, tags, min_quality) -> dict:
    """3-Stage Hybrid Filtering Pipeline:

    Stage 1 (Pre-filter): page_range, sections -> set of allowed chunk_ids
    Stage 2 (Hybrid search): vector (dense) + BM25 (sparse), both restricted to the
                             allowed chunk_ids, fused with Reciprocal Rank Fusion
    Stage 3 (Post-filter + Rerank): word_count, tags, quality -> cross-encoder.
                             The top reranker score is the confidence; below
                             MIN_RERANK_SCORE the answer is "not found".
    """
    doc_store = document_stores[doc_id]
    t_start = time.time()

    stages = {
        "stage1_prefilter_ms": 0,
        "stage1_chunks_after": len(doc_store.chunks),
        "stage2_search_ms": 0,
        "stage2_candidates": 0,
        "stage3_postfilter_ms": 0,
        "stage3_rerank_ms": 0,
        "stage3_final_docs": 0,
    }

    # ===== STAGE 1: Pre-filter (indexed, selective) =====
    has_prefilter = any([page_start, page_end, sections])
    allowed_ids = None

    if has_prefilter:
        allowed_ids = doc_store.metadata_index.pre_filter(
            page_start=page_start, page_end=page_end, sections=sections,
        )
        if allowed_ids is not None:
            stages["stage1_chunks_after"] = len(allowed_ids)
            if not allowed_ids:
                return _empty_result("No chunks match the specified filters.", t_start, stages)

    t_prefilter = time.time()
    prefilter_ms = (t_prefilter - t_start) * 1000
    stages["stage1_prefilter_ms"] = round(prefilter_ms, 2)

    # ===== STAGE 2: Hybrid search (dense + BM25, fused with RRF) =====
    query_vec = models.embeddings.embed_query(question)
    dense_ids = _dense_search(doc_store, question, allowed_ids, DENSE_TOP_K, query_vec)
    sparse_ids = _bm25_search(doc_store, question, allowed_ids, BM25_TOP_K)
    scope = None
    if TWO_STAGE and doc_store.section_index is not None:
        scope = _section_scope(doc_store.section_index, question, query_vec)
        if allowed_ids is not None:
            scope = scope & allowed_ids
    if scope:
        candidate_ids = _candidate_pool(_dense_search(doc_store, question, scope, DENSE_TOP_K, query_vec),
                                        _bm25_search(doc_store, question, scope, BM25_TOP_K))
        for chunk_id in _rrf_fuse([dense_ids, sparse_ids])[:GLOBAL_EXTRA]:   # right section may be missed
            if chunk_id not in candidate_ids:
                candidate_ids.append(chunk_id)
        stages["stage2_scope_chunks"] = len(scope)
    else:
        candidate_ids = _candidate_pool(dense_ids, sparse_ids)

    t_search = time.time()
    search_ms = (t_search - t_prefilter) * 1000
    stages["stage2_search_ms"] = round(search_ms, 2)
    stages["stage2_candidates"] = len(candidate_ids)
    stages["stage2_dense_hits"] = len(dense_ids)
    stages["stage2_bm25_hits"] = len(sparse_ids)

    if not candidate_ids:
        return _empty_result("No relevant content found in the document.", t_start, stages)

    # ===== STAGE 3: Post-filter + Rerank =====
    has_postfilter = any([min_word_count, tags, min_quality])
    stages["stage3_postfilter_relaxed"] = False

    if has_postfilter:
        kept_ids = doc_store.metadata_index.post_filter(
            candidate_ids,
            min_word_count=min_word_count,
            tags=tags,
            min_quality=min_quality,
        )
        if kept_ids:
            candidate_ids = kept_ids
        else:
            # Post-filter would remove everything: keep the unfiltered candidates
            stages["stage3_postfilter_relaxed"] = True

    is_broad = _is_broad_question(question)
    use_context = is_broad and RERANK_WITH_CONTEXT
    rerank_source = doc_store.indexed_chunks if use_context else doc_store.chunks
    candidate_docs = [rerank_source[i] for i in candidate_ids]

    t_postfilter = time.time()
    postfilter_ms = (t_postfilter - t_search) * 1000
    stages["stage3_postfilter_ms"] = round(postfilter_ms, 2)

    top_n = 7 if is_broad else 5
    # Back to the plain chunks (without the context line) for everything the user sees
    ranked = [(doc_store.chunks[d.metadata["chunk_id"]], score)
              for d, score in models.rerank_documents(question, candidate_docs, top_n=top_n)]
    t_rerank = time.time()
    rerank_ms = (t_rerank - t_postfilter) * 1000
    stages["stage3_rerank_ms"] = round(rerank_ms, 2)
    stages["stage3_final_docs"] = len(ranked)

    if not ranked:
        return _empty_result("No relevant content found.", t_start, stages)

    reranked_docs = [doc for doc, _ in ranked]
    confidence = ranked[0][1]  # top cross-encoder relevance, 0-1

    # ===== Not-found gate =====
    if confidence < (MIN_RERANK_SCORE_BROAD if use_context else MIN_RERANK_SCORE):
        related = _build_structured_answer(question, ranked, doc_store)
        return _empty_result(
            "I couldn't find a direct answer to this in the document. "
            "The sections below are the closest matches, but they are only loosely related; "
            "try rephrasing around what the document covers.",
            t_start, stages, method="not_found", score=confidence,
            source_chunks=[_visible(doc.page_content)[:500] for doc in reranked_docs[:3]],
            answer_blocks=related["answer_blocks"],
        )

    # ===== Answer Generation =====
    qa_score = None
    if is_broad:
        result_data = _build_structured_answer(question, ranked, doc_store, broad=True)
    else:
        context = "\n\n".join(doc.page_content for doc in reranked_docs[:5])
        max_context_chars = 3000
        if len(context) > max_context_chars:
            context = context[:max_context_chars]

        qa_result = models.qa_reader({"question": question, "context": context})
        qa_score = round(float(qa_result["score"]), 4)

        if qa_result["score"] < 0.10:
            result_data = _build_structured_answer(question, ranked, doc_store)
        else:
            result_data = {"answer": qa_result["answer"], "method": "extractive_qa"}

    t_end = time.time()
    total_time_ms = (t_end - t_start) * 1000

    # Update metrics
    perf_metrics["total_queries"] += 1
    perf_metrics["query_times"].append(total_time_ms)
    perf_metrics["avg_retrieval_time_ms"] = round(
        sum(perf_metrics["query_times"]) / len(perf_metrics["query_times"]), 2)
    if has_prefilter:
        perf_metrics["prefilter_times"].append(prefilter_ms)
    if has_postfilter:
        perf_metrics["postfilter_times"].append(postfilter_ms)

    logger.info(
        f"Query [{username}] in {total_time_ms:.0f}ms | "
        f"Pre:{prefilter_ms:.1f}ms Search:{search_ms:.1f}ms "
        f"Post:{postfilter_ms:.1f}ms Rerank:{rerank_ms:.1f}ms | "
        f"method={result_data.get('method')} confidence={confidence:.3f}")

    return {
        "answer": result_data["answer"],
        "answer_blocks": result_data.get("answer_blocks"),
        "score": round(confidence, 4),
        "qa_score": qa_score,
        "answer_found": result_data.get("method") != "none",
        "method": result_data.get("method", "extractive"),
        "source_chunks": [_visible(doc.page_content)[:500] for doc in reranked_docs[:3]],
        "retrieval_pipeline": RETRIEVAL_PIPELINE_LABEL,
        "pipeline_stages": stages,
        "retrieval_time_ms": round((t_rerank - t_start) * 1000, 2),
        "total_time_ms": round(total_time_ms, 2),
    }


# ---------------------------------------------------------------------------
# GET /doc-info/{doc_id} — Get document metadata and available filters
# ---------------------------------------------------------------------------

@app.get("/doc-info/{doc_id}")
async def get_doc_info(doc_id: str, user: dict = Depends(get_current_user)):
    doc_store = _owned_store(doc_id, user)
    meta_idx = doc_store.metadata_index

    # Collect all unique tags across chunks
    all_tags = set()
    for chunk_meta in meta_idx.chunks:
        all_tags.update(chunk_meta.tags)

    return {
        "doc_id": doc_id,
        "filename": doc_store.doc_info["filename"],
        "num_chunks": doc_store.doc_info["num_chunks"],
        "num_pages": doc_store.doc_info["num_pages"],
        "total_pages": meta_idx.total_pages,
        "sections": meta_idx.all_sections,
        "available_tags": sorted(all_tags),
        "filters_available": {
            "pre_filter": ["page_range", "sections"],
            "post_filter": ["min_word_count", "tags", "min_quality"],
        },
    }


# ---------------------------------------------------------------------------
# GET /health/
# ---------------------------------------------------------------------------

@app.get("/health/")
async def health():
    return {
        "status": "ok",
        "models_status": models.status,
        "models_ready": models.is_ready,
        "loaded_documents": len(document_stores),
        "total_chunks_indexed": perf_metrics["total_chunks_indexed"],
        "total_queries_served": perf_metrics["total_queries"],
        "avg_query_time_ms": perf_metrics["avg_retrieval_time_ms"],
        "architecture": "3-Stage Hybrid Filtering",
        "stages": {
            "stage1": "Pre-filter (indexed: page_range, sections) — O(1) lookup",
            "stage2": "Hybrid search: vectors (BGE-large-en-v1.5) + BM25, fused with RRF",
            "stage3": "Post-filter (word_count, tags, quality) + CrossEncoder Rerank",
        },
    }


# ---------------------------------------------------------------------------
# GET /stats/
# ---------------------------------------------------------------------------

@app.get("/stats/")
async def stats(user: dict = Depends(get_current_user)):
    return {
        "total_uploads": perf_metrics["total_uploads"],
        "total_queries": perf_metrics["total_queries"],
        "total_chunks_indexed": perf_metrics["total_chunks_indexed"],
        "avg_query_time_ms": perf_metrics["avg_retrieval_time_ms"],
        "avg_embedding_time_ms": perf_metrics["avg_embedding_time_ms"],
        "avg_prefilter_time_ms": round(
            sum(perf_metrics["prefilter_times"]) / max(len(perf_metrics["prefilter_times"]), 1), 2),
        "avg_postfilter_time_ms": round(
            sum(perf_metrics["postfilter_times"]) / max(len(perf_metrics["postfilter_times"]), 1), 2),
        "models_status": models.status,
        "documents": {k: v for k, v in doc_metadata.items() if v.get("uploaded_by") == user["username"]},
    }
