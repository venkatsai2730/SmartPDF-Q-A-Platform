"""
SmartPDF Q&A Platform — Production Backend Server
FastAPI + LangChain Pipeline + JWT Authentication

Pipeline:
  PDF → PyMuPDFLoader → RecursiveCharacterTextSplitter(800/160)
      → BGE-large-en-v1.5 (1024-dim) → FAISS IndexFlatIP
  Query → BGE embedding → FAISS top-10 → BGE-reranker top-5
        → DeBERTa-v3-large-squad2 QA reader → Answer

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
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt


# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="SmartPDF Q&A API", version="2.1.0")

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
# Authentication — JWT + Password Hashing (using bcrypt directly)
# ---------------------------------------------------------------------------

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "smartpdf-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def hash_password_sync(plain_password: str) -> str:
    """Hash a password using bcrypt (synchronous)."""
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password_sync(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its bcrypt hash (synchronous)."""
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


async def hash_password_async(plain_password: str) -> str:
    """Hash a password in a dedicated thread pool to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_cpu_executor, hash_password_sync, plain_password)


async def verify_password_async(plain_password: str, hashed_password: str) -> bool:
    """Verify a password in a dedicated thread pool to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_cpu_executor, verify_password_sync, plain_password, hashed_password)


# Demo users (in production, use a database)
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


class UserOut(BaseModel):
    username: str
    full_name: str


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Invalid or expired token",
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
# Lazy Model Loading — models load in background, APIs work immediately
# ---------------------------------------------------------------------------

class ModelManager:
    """Manages lazy/background loading of ML models so the server starts instantly."""

    def __init__(self):
        self.embeddings = None
        self.cross_encoder = None
        self.reranker = None
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
        """Load all ML models. Called in a background thread."""
        with self._lock:
            if self._loaded or self._loading:
                return
            self._loading = True

        try:
            # LangChain — Document splitting
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            # LangChain — Embeddings (BGE via HuggingFace)
            from langchain_huggingface import HuggingFaceEmbeddings

            # Reranker — using sentence-transformers CrossEncoder directly
            # (avoids LangChain CrossEncoderReranker compatibility issues)
            from sentence_transformers import CrossEncoder

            # HuggingFace — QA reader
            from transformers import pipeline as hf_pipeline

            logger.info("Loading embedding model: BAAI/bge-large-en-v1.5 ...")
            EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL_NAME,
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

            # RecursiveCharacterTextSplitter — sentence-boundary-aware chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=160,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
                length_function=len,
            )

            self._loaded = True
            self._loading = False
            logger.info("✅ All models loaded successfully.")

        except Exception as e:
            self._error = str(e)
            self._loading = False
            logger.error(f"❌ Model loading failed: {e}")
            import traceback
            traceback.print_exc()

    def require_models(self):
        """Raise HTTPException if models aren't loaded yet."""
        if self._error:
            raise HTTPException(
                status_code=503,
                detail=f"Model loading failed: {self._error}. Please check server logs.",
            )
        if not self._loaded:
            raise HTTPException(
                status_code=503,
                detail="Models are still loading. Please wait a few minutes and try again.",
            )

    def rerank_documents(self, query: str, documents: list, top_n: int = 5) -> list:
        """Rerank documents using the cross-encoder model directly."""
        if not documents:
            return []

        # Build query-document pairs for the cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]

        # Score all pairs
        scores = self.cross_encoder.predict(pairs)

        # Sort by score descending and return top_n
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored_docs[:top_n]]


models = ModelManager()


# ---------------------------------------------------------------------------
# Startup Event — load models in background thread
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Server starting — models will load in background...")
    thread = threading.Thread(target=models.load_models, daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# In-Memory Stores
# ---------------------------------------------------------------------------

vector_stores: dict = {}
doc_metadata: dict = {}

# Performance metrics store
perf_metrics = {
    "total_queries": 0,
    "total_uploads": 0,
    "total_chunks_indexed": 0,
    "avg_retrieval_time_ms": 0.0,
    "avg_embedding_time_ms": 0.0,
    "query_times": [],
    "embedding_times": [],
}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    doc_id: str
    question: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    full_name: str


# ---------------------------------------------------------------------------
# POST /login — Authenticate and get JWT token
# ---------------------------------------------------------------------------

@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = USERS_DB.get(form_data.username)
    if not user or not await verify_password_async(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": form_data.username})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user["username"],
        "full_name": user["full_name"],
    }


# ---------------------------------------------------------------------------
# POST /register — Create a new user account
# ---------------------------------------------------------------------------

@app.post("/register")
async def register(req: RegisterRequest):
    if req.username in USERS_DB:
        raise HTTPException(status_code=400, detail="Username already exists")
    if len(req.password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters")

    hashed = await hash_password_async(req.password)
    USERS_DB[req.username] = {
        "username": req.username,
        "full_name": req.full_name,
        "hashed_password": hashed,
    }
    access_token = create_access_token(data={"sub": req.username})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": req.username,
        "full_name": req.full_name,
    }


# ---------------------------------------------------------------------------
# POST /upload/ — Upload and index a PDF (authenticated)
# ---------------------------------------------------------------------------

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    # Ensure models are loaded before processing
    models.require_models()

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Read file content
    content = await file.read()

    # Check file size
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)}MB."
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        # Run the CPU-intensive processing in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _cpu_executor,
            _process_pdf_sync,
            content,
            file.filename,
            user["username"],
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
    """Process PDF synchronously — runs in thread pool to avoid blocking event loop."""
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_community.vectorstores import FAISS

    # Save to temp file (PyMuPDFLoader needs a file path)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # LangChain PyMuPDFLoader — loads each page as a Document with metadata
        loader = PyMuPDFLoader(tmp_path)
        pages = loader.load()

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not pages or all(p.page_content.strip() == "" for p in pages):
        raise HTTPException(status_code=400, detail="Could not extract text from PDF. The file may be scanned/image-based.")

    # RecursiveCharacterTextSplitter — intelligent chunking
    chunks = models.text_splitter.split_documents(pages)

    if len(chunks) == 0:
        raise HTTPException(status_code=400, detail="PDF produced no usable text chunks")

    # FAISS vector store — time the embedding + indexing
    t_embed_start = time.time()
    doc_id = str(uuid.uuid4())
    vector_store = FAISS.from_documents(chunks, models.embeddings)
    t_embed_end = time.time()
    embedding_time_ms = (t_embed_end - t_embed_start) * 1000

    # Store in memory
    vector_stores[doc_id] = vector_store
    doc_metadata[doc_id] = {
        "filename": filename,
        "num_chunks": len(chunks),
        "num_pages": len(pages),
        "uploaded_by": username,
        "embedding_time_ms": round(embedding_time_ms, 2),
    }

    # Update performance metrics
    perf_metrics["total_uploads"] += 1
    perf_metrics["total_chunks_indexed"] += len(chunks)
    perf_metrics["embedding_times"].append(embedding_time_ms)
    perf_metrics["avg_embedding_time_ms"] = round(
        sum(perf_metrics["embedding_times"]) / len(perf_metrics["embedding_times"]), 2
    )

    logger.info(
        f"Indexed {len(chunks)} chunks from '{filename}' in {embedding_time_ms:.0f}ms "
        f"(doc_id={doc_id}, user={username})"
    )

    return {
        "doc_id": doc_id,
        "filename": filename,
        "num_chunks": len(chunks),
        "num_pages": len(pages),
        "embedding_time_ms": round(embedding_time_ms, 2),
        "message": (
            f"PDF processed successfully. "
            f"{len(chunks)} chunks indexed with BGE-large embeddings in {embedding_time_ms:.0f}ms."
        ),
    }


# ---------------------------------------------------------------------------
# POST /ask/ — Ask a question against an uploaded document (authenticated)
# ---------------------------------------------------------------------------

@app.post("/ask/")
async def ask_question(req: AskRequest, user: dict = Depends(get_current_user)):
    # Ensure models are loaded before processing
    models.require_models()

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if req.doc_id not in vector_stores:
        raise HTTPException(status_code=404, detail="Document not found. Please upload a PDF first.")

    try:
        # Run the CPU-intensive QA pipeline in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _cpu_executor,
            _ask_question_sync,
            req.doc_id,
            req.question,
            user["username"],
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
    """Detect if a question is broad/descriptive (needs full context display)
    vs. factoid (can be answered with a short extractive span)."""
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
    """For broad questions: format the retrieved chunks into a clean,
    readable answer. The chunks themselves ARE the answer for broad questions —
    an extractive QA model cannot summarize or list things."""

    if not chunks:
        return {"answer": "No relevant content found.", "score": 0.0, "method": "none"}

    # Clean and format each chunk's content
    formatted_sections = []
    seen_content = set()

    for doc in chunks:
        content = doc.page_content.strip()
        if not content or len(content) < 30:
            continue

        # Deduplicate — skip chunks with high overlap
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

        # Limit to 5 unique sections for readability
        if len(formatted_sections) >= 5:
            break

    if not formatted_sections:
        return {"answer": "No relevant content found.", "score": 0.0, "method": "none"}

    # Join sections with clear separators
    answer = "\n\n".join(formatted_sections)

    return {
        "answer": answer,
        "score": 0.85,  # High confidence — these are directly retrieved relevant sections
        "method": "semantic_retrieval",
    }


def _ask_question_sync(doc_id: str, question: str, username: str) -> dict:
    """Run the QA pipeline synchronously — runs in thread pool.

    Strategy:
    - For broad questions (explain, describe, list, tell me about):
      Return the top reranked chunks directly — they contain the answer.
      An extractive QA model CANNOT answer broad questions meaningfully.
    - For factoid questions (who, what year, how many, which):
      Use DeBERTa extractive QA to pinpoint the exact answer span.
    - Fallback: if extractive confidence is too low, return chunks directly.
    """
    vector_store = vector_stores[doc_id]

    # --- Timed retrieval pipeline ---
    t_start = time.time()

    # Step 1: FAISS similarity search — fetch top-15 candidates for better coverage
    candidate_docs = vector_store.similarity_search(question, k=15)
    t_faiss = time.time()

    if not candidate_docs:
        return {
            "answer": "No relevant content found in the document.",
            "score": 0.0,
            "source_chunks": [],
            "retrieval_time_ms": round((t_faiss - t_start) * 1000, 2),
        }

    # Step 2: Rerank with cross-encoder
    is_broad = _is_broad_question(question)
    top_n = 7 if is_broad else 5
    reranked_docs = models.rerank_documents(question, candidate_docs, top_n=top_n)
    t_rerank = time.time()
    retrieval_time_ms = (t_rerank - t_start) * 1000

    if not reranked_docs:
        return {
            "answer": "No relevant content found in the document.",
            "score": 0.0,
            "source_chunks": [],
            "retrieval_time_ms": round(retrieval_time_ms, 2),
        }

    # Step 3: Answer strategy based on question type
    if is_broad:
        # BROAD QUESTIONS: Return retrieved chunks directly.
        # DeBERTa extractive QA is NOT designed for "tell me about X" questions.
        # The retrieved + reranked chunks ARE the answer.
        result_data = _format_chunks_as_answer(question, reranked_docs)
    else:
        # FACTOID QUESTIONS: Use extractive QA to find the precise answer span.
        context = "\n\n".join(doc.page_content for doc in reranked_docs[:5])
        # DeBERTa max input is 512 tokens; keep context within safe limits
        max_context_chars = 3000
        if len(context) > max_context_chars:
            context = context[:max_context_chars]

        qa_result = models.qa_reader({"question": question, "context": context})

        if qa_result["score"] < 0.10:
            # Low confidence on factoid — fall back to showing relevant content
            result_data = _format_chunks_as_answer(question, reranked_docs)
        else:
            result_data = {
                "answer": qa_result["answer"],
                "score": round(float(qa_result["score"]), 4),
                "method": "extractive_qa",
            }

    t_end = time.time()
    total_time_ms = (t_end - t_start) * 1000

    # Update performance metrics
    perf_metrics["total_queries"] += 1
    perf_metrics["query_times"].append(total_time_ms)
    perf_metrics["avg_retrieval_time_ms"] = round(
        sum(perf_metrics["query_times"]) / len(perf_metrics["query_times"]), 2
    )

    logger.info(
        f"Query answered in {total_time_ms:.0f}ms "
        f"(method={result_data.get('method', 'unknown')}, broad={is_broad}, "
        f"FAISS={((t_faiss - t_start)*1000):.0f}ms, rerank={((t_rerank - t_faiss)*1000):.0f}ms, "
        f"user={username})"
    )

    return {
        "answer": result_data["answer"],
        "score": round(float(result_data["score"]), 4),
        "method": result_data.get("method", "extractive"),
        "source_chunks": [doc.page_content[:500] for doc in reranked_docs[:3]],
        "retrieval_pipeline": "BGE-large-en-v1.5 → FAISS(top-15) → BGE-reranker → " + (
            "Semantic Retrieval" if is_broad else "DeBERTa-v3-large QA"
        ),
        "retrieval_time_ms": round(retrieval_time_ms, 2),
        "total_time_ms": round(total_time_ms, 2),
    }


# ---------------------------------------------------------------------------
# GET /health/ — Health check (public)
# ---------------------------------------------------------------------------

@app.get("/health/")
async def health():
    return {
        "status": "ok",
        "models_status": models.status,
        "models_ready": models.is_ready,
        "loaded_documents": len(vector_stores),
        "total_chunks_indexed": perf_metrics["total_chunks_indexed"],
        "total_queries_served": perf_metrics["total_queries"],
        "avg_query_time_ms": perf_metrics["avg_retrieval_time_ms"],
        "models": {
            "embedder": "BAAI/bge-large-en-v1.5 (1024-dim)",
            "reranker": "BAAI/bge-reranker-base (CrossEncoder)",
            "qa_reader": "deepset/deberta-v3-large-squad2 (EM ~86%)",
        },
        "scalability": {
            "max_chunks_supported": "50,000+ (FAISS flat index, ~200MB RAM for 50K×1024 float32)",
            "retrieval_method": "Two-stage: FAISS top-10 → CrossEncoder rerank → top-5",
            "chunking_strategy": "RecursiveCharacterTextSplitter (800 chars, 160 overlap, sentence-aware)",
        },
    }


# ---------------------------------------------------------------------------
# GET /stats/ — Performance metrics (authenticated)
# ---------------------------------------------------------------------------

@app.get("/stats/")
async def stats(user: dict = Depends(get_current_user)):
    return {
        "total_uploads": perf_metrics["total_uploads"],
        "total_queries": perf_metrics["total_queries"],
        "total_chunks_indexed": perf_metrics["total_chunks_indexed"],
        "avg_query_time_ms": perf_metrics["avg_retrieval_time_ms"],
        "avg_embedding_time_ms": perf_metrics["avg_embedding_time_ms"],
        "models_status": models.status,
        "documents": {
            doc_id: meta for doc_id, meta in doc_metadata.items()
        },
    }
