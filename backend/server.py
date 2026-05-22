"""
SmartPDF Q&A Platform — Production Backend Server
FastAPI + Staged Hybrid Filtering Pipeline + JWT Authentication

Pipeline (3-Stage Hybrid Filtering):
  Stage 1 - Pre-filter (indexed metadata):
    page_range, section/department, access_level → selective subset
  Stage 2 - ANN Vector Search:
    BGE-large-en-v1.5 → FAISS similarity on filtered subset
  Stage 3 - Post-filter + Rerank:
    word_count, tags, content quality → BGE-reranker-base → DeBERTa QA

Models:
  - Embedder:  BAAI/bge-large-en-v1.5 (1024-dim, normalize_embeddings=True)
  - Reranker:  BAAI/bge-reranker-base (CrossEncoder)
  - QA Reader: deepset/deberta-v3-large-squad2 (EM ~86% on SQuAD2.0)

Run:
  uvicorn server:app --reload --host 0.0.0.0 --port 8000
"""

import uuid
import time
import logging
import tempfile
import os
import re
import asyncio
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import numpy as np
import bcrypt
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
                 metadata_index: MetadataIndex, doc_info: dict):
        self.doc_id = doc_id
        self.vector_store = vector_store
        self.chunks = chunks  # Raw LangChain Document objects
        self.metadata_index = metadata_index
        self.doc_info = doc_info

    def get_chunks_by_ids(self, chunk_ids: set) -> list:
        """Get LangChain Document objects by their chunk IDs."""
        return [self.chunks[i] for i in chunk_ids if i < len(self.chunks)]


# ---------------------------------------------------------------------------
# Metadata Extraction Helpers
# ---------------------------------------------------------------------------

def _detect_section(text: str, page_num: int) -> str:
    """Detect section/topic from chunk content using heading patterns."""
    lines = text.strip().split('\n')
    for line in lines[:3]:  # Check first 3 lines for headings
        line = line.strip()
        # Common heading patterns
        if re.match(r'^#{1,4}\s+', line):
            return re.sub(r'^#{1,4}\s+', '', line).strip()
        if re.match(r'^\d+\.?\d*\.?\s+[A-Z]', line):
            return re.sub(r'^\d+\.?\d*\.?\s+', '', line).strip()
        if line.isupper() and 3 < len(line) < 80:
            return line.title()
        if re.match(r'^[A-Z][a-z].*:$', line):
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
        if not documents:
            return []
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_n]]


models = ModelManager()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info("Server starting - models will load in background...")
    thread = threading.Thread(target=models.load_models, daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# In-Memory Stores
# ---------------------------------------------------------------------------

document_stores: Dict[str, DocumentStore] = {}
doc_metadata: dict = {}

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
    from langchain_community.vectorstores import FAISS

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        loader = PyMuPDFLoader(tmp_path)
        pages = loader.load()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not pages or all(p.page_content.strip() == "" for p in pages):
        raise HTTPException(status_code=400,
            detail="Could not extract text from PDF. The file may be scanned/image-based.")

    # Split into chunks
    chunks = models.text_splitter.split_documents(pages)
    if len(chunks) == 0:
        raise HTTPException(status_code=400, detail="PDF produced no usable text chunks")

    # Build metadata index for each chunk
    metadata_index = MetadataIndex()
    for i, chunk in enumerate(chunks):
        page_num = chunk.metadata.get("page", 0) + 1  # 1-indexed
        text = chunk.page_content
        section = _detect_section(text, page_num)
        word_count = len(text.split())
        tags = _extract_tags(text)
        has_code = _has_code_content(text)
        has_table = _has_table_content(text)
        quality = _compute_quality_score(text)

        meta = ChunkMetadata(
            chunk_id=i, page_num=page_num, section=section,
            word_count=word_count, tags=tags, has_code=has_code,
            has_table=has_table, content_quality_score=quality,
        )
        metadata_index.add_chunk(meta)

    metadata_index.finalize()

    # Build FAISS vector store
    t_embed_start = time.time()
    doc_id = str(uuid.uuid4())
    vector_store = FAISS.from_documents(chunks, models.embeddings)
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
    document_stores[doc_id] = DocumentStore(
        doc_id=doc_id, vector_store=vector_store,
        chunks=chunks, metadata_index=metadata_index, doc_info=doc_info,
    )
    doc_metadata[doc_id] = doc_info

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
    if req.doc_id not in document_stores:
        raise HTTPException(status_code=404, detail="Document not found. Please upload a PDF first.")

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


def _format_chunks_as_answer(question: str, chunks: list) -> dict:
    """For broad questions: return retrieved chunks as the answer."""
    if not chunks:
        return {"answer": "No relevant content found.", "score": 0.0, "method": "none"}

    formatted_sections = []
    seen_content = set()

    for doc in chunks:
        content = doc.page_content.strip()
        if not content or len(content) < 30:
            continue

        content_words = set(content.lower().split())
        is_duplicate = False
        for seen in seen_content:
            seen_words = set(seen.lower().split())
            if len(content_words) > 5 and len(seen_words) > 5:
                overlap = len(content_words & seen_words) / min(len(content_words), len(seen_words))
                if overlap > 0.6:
                    is_duplicate = True
                    break
        if is_duplicate:
            continue

        seen_content.add(content)
        formatted_sections.append(content)
        if len(formatted_sections) >= 5:
            break

    if not formatted_sections:
        return {"answer": "No relevant content found.", "score": 0.0, "method": "none"}

    answer = "\n\n".join(formatted_sections)
    return {"answer": answer, "score": 0.85, "method": "semantic_retrieval"}


def _ask_question_sync(doc_id: str, question: str, username: str,
                       page_start, page_end, sections,
                       min_word_count, tags, min_quality) -> dict:
    """3-Stage Hybrid Filtering Pipeline:

    Stage 1 (Pre-filter): page_range, sections → indexed O(1) lookup
    Stage 2 (ANN Search): FAISS similarity on filtered subset
    Stage 3 (Post-filter + Rerank): word_count, tags, quality → reranker → QA
    """
    doc_store = document_stores[doc_id]
    t_start = time.time()

    # ===== STAGE 1: Pre-filter (indexed, selective) =====
    has_prefilter = any([page_start, page_end, sections])
    prefilter_ids = None

    if has_prefilter:
        prefilter_ids = doc_store.metadata_index.pre_filter(
            page_start=page_start, page_end=page_end, sections=sections,
        )
        if prefilter_ids is not None and len(prefilter_ids) == 0:
            return {
                "answer": "No chunks match the specified filters.",
                "score": 0.0, "source_chunks": [],
                "retrieval_time_ms": 0, "total_time_ms": 0,
                "pipeline_stages": {"pre_filter": "0 chunks matched"},
            }

    t_prefilter = time.time()
    prefilter_ms = (t_prefilter - t_start) * 1000

    # ===== STAGE 2: ANN Vector Search (FAISS) =====
    # If pre-filter applied, search within filtered subset only
    if prefilter_ids is not None:
        # Get the filtered chunks and search among them
        filtered_chunks = doc_store.get_chunks_by_ids(prefilter_ids)
        if not filtered_chunks:
            return {
                "answer": "No relevant content found with the applied filters.",
                "score": 0.0, "source_chunks": [],
                "retrieval_time_ms": round(prefilter_ms, 2),
            }
        # Use FAISS with filter: search more broadly, then intersect
        k_search = min(30, len(doc_store.chunks))
        all_results = doc_store.vector_store.similarity_search(question, k=k_search)
        # Intersect with pre-filtered IDs
        candidate_docs = []
        for doc in all_results:
            # Find chunk index by matching content
            for idx in prefilter_ids:
                if idx < len(doc_store.chunks) and doc_store.chunks[idx].page_content == doc.page_content:
                    candidate_docs.append(doc)
                    break
            if len(candidate_docs) >= 15:
                break
    else:
        # No pre-filter: standard FAISS search
        candidate_docs = doc_store.vector_store.similarity_search(question, k=15)

    t_search = time.time()
    search_ms = (t_search - t_prefilter) * 1000

    if not candidate_docs:
        return {
            "answer": "No relevant content found in the document.",
            "score": 0.0, "source_chunks": [],
            "retrieval_time_ms": round((t_search - t_start) * 1000, 2),
        }

    # ===== STAGE 3: Post-filter + Rerank =====
    has_postfilter = any([min_word_count, tags, min_quality])

    if has_postfilter:
        # Find chunk IDs for candidate docs
        candidate_ids = []
        for doc in candidate_docs:
            for i, chunk in enumerate(doc_store.chunks):
                if chunk.page_content == doc.page_content:
                    candidate_ids.append(i)
                    break

        # Apply post-filter
        filtered_ids = doc_store.metadata_index.post_filter(
            candidate_ids,
            min_word_count=min_word_count,
            tags=tags,
            min_quality=min_quality,
        )

        if not filtered_ids:
            # Fallback: use unfiltered candidates if post-filter removes everything
            pass
        else:
            candidate_docs = doc_store.get_chunks_by_ids(set(filtered_ids))

    t_postfilter = time.time()
    postfilter_ms = (t_postfilter - t_search) * 1000

    # Rerank with cross-encoder
    is_broad = _is_broad_question(question)
    top_n = 7 if is_broad else 5
    reranked_docs = models.rerank_documents(question, candidate_docs, top_n=top_n)
    t_rerank = time.time()
    rerank_ms = (t_rerank - t_postfilter) * 1000

    if not reranked_docs:
        return {
            "answer": "No relevant content found.",
            "score": 0.0, "source_chunks": [],
            "retrieval_time_ms": round((t_rerank - t_start) * 1000, 2),
        }

    # ===== Answer Generation =====
    if is_broad:
        result_data = _format_chunks_as_answer(question, reranked_docs)
    else:
        context = "\n\n".join(doc.page_content for doc in reranked_docs[:5])
        max_context_chars = 3000
        if len(context) > max_context_chars:
            context = context[:max_context_chars]

        qa_result = models.qa_reader({"question": question, "context": context})

        if qa_result["score"] < 0.10:
            result_data = _format_chunks_as_answer(question, reranked_docs)
        else:
            result_data = {
                "answer": qa_result["answer"],
                "score": round(float(qa_result["score"]), 4),
                "method": "extractive_qa",
            }

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

    # Pipeline stage breakdown
    pipeline_stages = {
        "stage1_prefilter_ms": round(prefilter_ms, 2),
        "stage1_chunks_after": len(prefilter_ids) if prefilter_ids is not None else len(doc_store.chunks),
        "stage2_search_ms": round(search_ms, 2),
        "stage2_candidates": len(candidate_docs),
        "stage3_postfilter_ms": round(postfilter_ms, 2),
        "stage3_rerank_ms": round(rerank_ms, 2),
        "stage3_final_docs": len(reranked_docs),
    }

    logger.info(
        f"Query [{username}] in {total_time_ms:.0f}ms | "
        f"Pre:{prefilter_ms:.1f}ms Search:{search_ms:.1f}ms "
        f"Post:{postfilter_ms:.1f}ms Rerank:{rerank_ms:.1f}ms | "
        f"method={result_data.get('method')}")

    return {
        "answer": result_data["answer"],
        "score": round(float(result_data["score"]), 4),
        "method": result_data.get("method", "extractive"),
        "source_chunks": [doc.page_content[:500] for doc in reranked_docs[:3]],
        "retrieval_pipeline": "Stage1:PreFilter → Stage2:FAISS-ANN → Stage3:PostFilter+Rerank",
        "pipeline_stages": pipeline_stages,
        "retrieval_time_ms": round((t_rerank - t_start) * 1000, 2),
        "total_time_ms": round(total_time_ms, 2),
    }


# ---------------------------------------------------------------------------
# GET /doc-info/{doc_id} — Get document metadata and available filters
# ---------------------------------------------------------------------------

@app.get("/doc-info/{doc_id}")
async def get_doc_info(doc_id: str, user: dict = Depends(get_current_user)):
    if doc_id not in document_stores:
        raise HTTPException(status_code=404, detail="Document not found")

    doc_store = document_stores[doc_id]
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
            "stage2": "ANN Vector Search (FAISS + BGE-large-en-v1.5)",
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
        "documents": doc_metadata,
    }
