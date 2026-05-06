"""
SmartPDF Q&A Platform — Production Backend Server
FastAPI + LangChain LCEL Pipeline + JWT Authentication

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
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext

# LangChain — Document loading & splitting
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain — Embeddings (BGE via HuggingFace)
from langchain_huggingface import HuggingFaceEmbeddings

# LangChain — Vector store
from langchain_community.vectorstores import FAISS

# LangChain — Reranker (Cross-Encoder)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# HuggingFace — QA reader
from transformers import pipeline as hf_pipeline


# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="SmartPDF Q&A API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Authentication — JWT + Password Hashing
# ---------------------------------------------------------------------------

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "smartpdf-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Demo users (in production, use a database)
USERS_DB = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "hashed_password": pwd_context.hash("admin123"),
    },
    "demo": {
        "username": "demo",
        "full_name": "Demo User",
        "hashed_password": pwd_context.hash("demo123"),
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


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: timedelta = None):
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
# Model Initialization (loaded ONCE at module level — NOT inside endpoints)
# ---------------------------------------------------------------------------

logger.info("Loading embedding model: BAAI/bge-large-en-v1.5 ...")

EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

logger.info("Loading cross-encoder reranker: BAAI/bge-reranker-base ...")

cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

logger.info("Loading QA reader: deepset/deberta-v3-large-squad2 ...")

qa_reader = hf_pipeline(
    "question-answering",
    model="deepset/deberta-v3-large-squad2",
    tokenizer="deepset/deberta-v3-large-squad2",
)

logger.info("All models loaded successfully.")


# ---------------------------------------------------------------------------
# In-Memory Stores
# ---------------------------------------------------------------------------

vector_stores: dict[str, FAISS] = {}
doc_metadata: dict[str, dict] = {}

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
# Text Splitter Configuration
# ---------------------------------------------------------------------------

# RecursiveCharacterTextSplitter — sentence-boundary-aware chunking
# chunk_size=800, chunk_overlap=160 (20% overlap) — optimal for BGE-large-en-v1.5
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=160,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    length_function=len,
)


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
    if not user or not verify_password(form_data.password, user["hashed_password"]):
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

    USERS_DB[req.username] = {
        "username": req.username,
        "full_name": req.full_name,
        "hashed_password": pwd_context.hash(req.password),
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
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Save to temp file (PyMuPDFLoader needs a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # LangChain PyMuPDFLoader — loads each page as a Document with metadata
        loader = PyMuPDFLoader(tmp_path)
        pages = loader.load()
        os.unlink(tmp_path)

        if not pages or all(p.page_content.strip() == "" for p in pages):
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        # RecursiveCharacterTextSplitter — intelligent chunking
        chunks = text_splitter.split_documents(pages)

        if len(chunks) == 0:
            raise HTTPException(status_code=400, detail="PDF produced no usable text chunks")

        # FAISS vector store — time the embedding + indexing
        t_embed_start = time.time()
        doc_id = str(uuid.uuid4())
        vector_store = FAISS.from_documents(chunks, embeddings)
        t_embed_end = time.time()
        embedding_time_ms = (t_embed_end - t_embed_start) * 1000

        # Store in memory
        vector_stores[doc_id] = vector_store
        doc_metadata[doc_id] = {
            "filename": file.filename,
            "num_chunks": len(chunks),
            "num_pages": len(pages),
            "uploaded_by": user["username"],
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
            f"Indexed {len(chunks)} chunks from '{file.filename}' in {embedding_time_ms:.0f}ms "
            f"(doc_id={doc_id}, user={user['username']})"
        )

        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "num_chunks": len(chunks),
            "num_pages": len(pages),
            "embedding_time_ms": round(embedding_time_ms, 2),
            "message": (
                f"PDF processed successfully. "
                f"{len(chunks)} chunks indexed with BGE-large embeddings in {embedding_time_ms:.0f}ms."
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# ---------------------------------------------------------------------------
# POST /ask/ — Ask a question against an uploaded document (authenticated)
# ---------------------------------------------------------------------------

@app.post("/ask/")
async def ask_question(req: AskRequest, user: dict = Depends(get_current_user)):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if req.doc_id not in vector_stores:
        raise HTTPException(status_code=404, detail="Document not found. Please upload a PDF first.")

    try:
        vector_store = vector_stores[req.doc_id]

        # --- Timed retrieval pipeline ---
        t_start = time.time()

        # Step 1: FAISS retriever — fetch top-10 candidates
        base_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},
        )

        # Step 2: ContextualCompressionRetriever with BGE cross-encoder reranker
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever,
        )

        # Step 3: Retrieve and rerank
        reranked_docs = compression_retriever.invoke(req.question)
        t_retrieval = time.time()
        retrieval_time_ms = (t_retrieval - t_start) * 1000

        if not reranked_docs:
            return {
                "answer": "No relevant content found in the document.",
                "score": 0.0,
                "source_chunks": [],
                "retrieval_time_ms": round(retrieval_time_ms, 2),
            }

        # Step 4: Build context from top-5 reranked chunks
        context = "\n\n".join(doc.page_content for doc in reranked_docs[:5])

        # Step 5: DeBERTa-v3-large QA reader
        result = qa_reader({"question": req.question, "context": context})
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
            f"(retrieval={retrieval_time_ms:.0f}ms, user={user['username']})"
        )

        return {
            "answer": result["answer"],
            "score": round(result["score"], 4),
            "source_chunks": [doc.page_content[:400] for doc in reranked_docs[:3]],
            "retrieval_pipeline": "BGE-large-en-v1.5 → FAISS → BGE-reranker-base → DeBERTa-v3-large",
            "retrieval_time_ms": round(retrieval_time_ms, 2),
            "total_time_ms": round(total_time_ms, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ask error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ---------------------------------------------------------------------------
# GET /health/ — Health check (public)
# ---------------------------------------------------------------------------

@app.get("/health/")
async def health():
    return {
        "status": "ok",
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
        "documents": {
            doc_id: meta for doc_id, meta in doc_metadata.items()
        },
    }
