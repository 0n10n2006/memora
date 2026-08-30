"""
MEMORA API
==========

Thin HTTP layer over the existing backend pipeline. Nothing in
here reimplements ingestion/retrieval/reconstruction logic — it
only calls MemoryPipeline, which itself wraps the ingestion,
indexing, relationship, and reconstruction modules that already
exist.

Run with:

    uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

Run from the MEMORA/ project root (same place you run the other
scripts from), so the `backend.*` imports and the relative
`data/` paths resolve the same way they already do.

Endpoints
---------
GET  /health                      liveness check
POST /ingest                      upload a file, fully process it
POST /remember                    ask a natural-language question
GET  /memories                    list all stored memories (summary)
GET  /memory/{memory_id}          full detail for one memory
GET  /memory/{memory_id}/relationships   just its relationships
"""

import shutil
import tempfile
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.reconstruction.memory_pipeline import (
    MemoryPipeline,
    MemoryIngestionError,
)


app = FastAPI(
    title="MEMORA API",
    description="Personal memory system backend.",
    version="0.1.0",
)

# CORS is wide open here on purpose: this is a same-weekend
# hackathon prototype talked to by an Android app / Office Kit
# bridge on the same local network, not a public deployment.
# Tighten this before shipping anything beyond the demo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The pipeline loads several models (embedder, semantic
# analyzer, reranker, reconstructor) at construction time.
# That takes real time and real memory, so it happens exactly
# once, when the server process starts — not per-request.
print("[MEMORA API] Loading pipeline (this can take a while)...")
pipeline = MemoryPipeline()
print("[MEMORA API] Pipeline loaded. Ready for requests.")

# MemoryStore._save() writes the entire memories.json in one
# write_text() call with no file locking. FastAPI runs sync
# routes in a thread pool, so two /ingest requests arriving at
# the same time could interleave: read-modify-write A, read-
# modify-write B, and B's write silently drops A's memory.
# A single lock around the whole ingest flow is the simplest
# fix that doesn't touch MemoryStore itself — it just means
# uploads process one at a time, which is fine for a hackathon
# demo (you're not expecting concurrent uploads from a crowd).
_ingest_lock = threading.Lock()


# =========================================================
# Request/response schemas
# =========================================================

class RememberRequest(BaseModel):
    query: str


# =========================================================
# Helpers
# =========================================================

def memory_to_dict(memory_object):
    """
    Convert a MemoryObject (and its nested MemoryRelationship
    dataclasses) into plain JSON-serializable dicts.
    """
    if memory_object is None:
        return None
    return asdict(memory_object)


# =========================================================
# Health
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "memories_stored": pipeline.store.count(),
    }


# =========================================================
# Ingest a new file
# =========================================================

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Accepts a file upload, runs it through the full pipeline
    (ingest -> semantic understanding -> index -> relate),
    and returns the resulting memory plus any relationships
    discovered against existing memories.
    """

    suffix = Path(file.filename).suffix

    # MemoryPipeline.add_memory() takes a path on disk, so the
    # uploaded bytes are written to a temp file first, using the
    # ORIGINAL filename as the stem: ingestion uses the filename
    # stem as the memory's id (see ingestion/parser.py and
    # ingestion/image_parser.py), so preserving it here keeps
    # ids meaningful instead of becoming a random temp name.
    tmp_dir = Path(tempfile.mkdtemp(prefix="memora_upload_"))
    tmp_path = tmp_dir / file.filename

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        started = time.perf_counter()

        with _ingest_lock:
            result = pipeline.add_memory(tmp_path)

        elapsed = round(time.perf_counter() - started, 2)

        print(f"[MEMORA API] /ingest '{file.filename}' took {elapsed}s")

        return {
            "id": result["id"],
            "chunks": result["chunks"],
            "relationships_found": len(result["relationships"]),
            "memory": memory_to_dict(result["memory"]),
            "processing_seconds": elapsed,
        }

    except MemoryIngestionError as error:
        # File was parsed but produced no usable content
        # (e.g. scanned PDF with no text layer). Nothing was
        # stored — this is a client-fixable problem, not a
        # server error, hence 422 rather than 500.
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest '{file.filename}': {error}",
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =========================================================
# Remember (query)
# =========================================================

@app.post("/remember")
def remember(request: RememberRequest):
    """
    Runs relationship-aware retrieval + reconstruction for a
    natural-language query and returns an answer with evidence.
    """

    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query must not be empty.",
        )

    started = time.perf_counter()

    try:
        result = pipeline.remember(request.query)
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {error}",
        )

    elapsed = round(time.perf_counter() - started, 2)

    print(f"[MEMORA API] /remember took {elapsed}s")

    reconstruction = result["reconstruction"]

    return {
        "query": result["query"],
        "answer": reconstruction["answer"],
        "confidence": reconstruction["confidence"],
        "evidence": reconstruction["evidence"],
        "processing_seconds": elapsed,
        "memories": [
            {
                "memory_id": memory.get("metadata", {}).get("memory_id"),
                "source": memory.get("metadata", {}).get("source"),
                "title": memory.get("metadata", {}).get("title"),
                "retrieval_type": memory.get("retrieval_type", "unknown"),
            }
            for memory in result["memories"]
        ],
    }


# =========================================================
# List all memories
# =========================================================

@app.get("/memories")
def list_memories():
    pipeline.store._load()
    memories = pipeline.store.all()

    return {
        "count": len(memories),
        "memories": [
            {
                "id": memory.id,
                "source": memory.source,
                "modality": memory.modality,
                "title": memory.title,
                "summary": memory.summary,
                "topics": memory.topics,
                "relationship_count": len(memory.relationships),
            }
            for memory in memories
        ],
    }


# =========================================================
# Single memory detail
# =========================================================

@app.get("/memory/{memory_id}")
def get_memory(memory_id: str):
    pipeline.store._load()
    memory = pipeline.store.get(memory_id)

    if not memory:
        raise HTTPException(
            status_code=404,
            detail=f"No memory found with id '{memory_id}'.",
        )

    return memory_to_dict(memory)


# =========================================================
# Single memory's relationships
# =========================================================

@app.get("/memory/{memory_id}/relationships")
def get_memory_relationships(memory_id: str):
    pipeline.store._load()
    memory = pipeline.store.get(memory_id)

    if not memory:
        raise HTTPException(
            status_code=404,
            detail=f"No memory found with id '{memory_id}'.",
        )

    relationships = pipeline.store.get_relationships(memory_id)

    return {
        "memory_id": memory_id,
        "relationships": [asdict(r) for r in relationships],
    }