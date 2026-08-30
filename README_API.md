# MEMORA API — what changed and how to use it

## Files to drop into your repo

```
backend/reconstruction/memory_pipeline.py   (updated)
backend/memory/relationship_engine.py       (updated — added one method)
backend/ingestion/ingestor.py               (updated — vision wiring fix)
backend/ingestion/parser.py                 (updated — .txt encoding fallback)
backend/api/main.py                         (new)
requirements-api.txt                        (new)
```

No existing function signatures were removed or changed. `batch.py`,
`indexer.py`'s CLI, `search.py`'s CLI, `image_parser.py`'s CLI, and
everything else still run exactly as before — the changes are
additive (new optional parameters, new methods, new exception type).

## Bugs found and fixed in this pass

**1. Vision (BLIP) never ran in the real pipeline.**
`ingestor.py` called `parse_image(path)` with no analyzer arguments,
so every image going through `ingest_file()` — which is what
`batch.py` and the API both use — skipped BLIP entirely and only did
OCR. Vision only ran if you manually instantiated `VisionAnalyzer`
and ran `image_parser.py` as a standalone CLI script. Fixed with a
lazy singleton in `ingestor.py` (`_get_vision_analyzer()`), loaded
once per process and reused. `MemoryPipeline.__init__` now also
warms it up at server startup instead of on the first demo image, so
you don't eat a multi-second model-load stall mid-pitch.
Set `MEMORA_SKIP_VISION=1` to disable it again if it slows things
down more than it's worth for your demo.

**2. Scanned/image-only PDFs silently vanished.**
`extract_pdf()` only reads existing text layers (no OCR fallback for
PDFs, unlike images). When a PDF had no text layer, `indexer.
index_memory()` returned 0 chunks *without ever calling
`memory_store.add()`* — no error, the file just disappeared. Fixed:
`MemoryPipeline.add_memory()` now raises a `MemoryIngestionError`
with a clear explanation in that case, and the API returns a `422`
with that message instead of a misleading `200 success` with
`memory: null`.

**3. `.txt` files assumed UTF-8 with no fallback.**
Any file in a different encoding raised an uncaught
`UnicodeDecodeError`. `extract_txt()` now tries utf-8 → utf-8-sig →
cp1252 → latin-1 in order (latin-1 always succeeds, so this can no
longer crash on encoding alone).

**4. Concurrent uploads could corrupt `memories.json`.**
`MemoryStore._save()` writes the whole JSON file with no locking.
FastAPI runs sync routes in a thread pool, so two simultaneous
`/ingest` calls could interleave writes and silently drop one.
Fixed with a single `threading.Lock()` around the ingest flow in
`main.py` only — `MemoryStore` itself wasn't touched. Uploads now
process one at a time, which is fine for a demo.

**5. Speed was unmeasured.**
Both `/ingest` and `/remember` now return a `processing_seconds`
field and log timing to the console, so you have real numbers
instead of guessing. I did not touch model generation parameters
(beam counts, max tokens) — changing those affects output quality
and wasn't worth the risk this close to a demo. If a real number
comes back too slow, the honest options are: reduce BLIP's 3
beam-search questions per image to 1, or reduce Qwen's
`max_new_tokens`, or move to GPU if the venue's laptop has one. None
of that is done here — measure first, then decide if it's needed.

## What's still NOT fixed, on purpose

- No OCR fallback for scanned PDFs (only for images). Fixing this
  well means rasterizing PDF pages to images and running Tesseract
  per page — a bigger change than a day allows safely. Workaround:
  convert scanned PDFs to images before uploading, or just avoid
  them in your demo set.
- `MemoryStore` still has no `delete()` — out of scope, matches your
  own "don't keep rewriting memory_store.py" note.
- Phone-first / NPU inference: nothing here runs on-device. That's
  a hardware/deployment question, not something a code patch to the
  backend logic solves.

## What was actually missing

1. There was no single call that takes **one new file** and makes it
   fully searchable + related. You had three separate manual scripts
   (`ingestor.py`, `indexer.py`, `relationship_engine.py`'s full
   `discover_relationships()` which recomputes *every* pair in the
   whole store from scratch every time). `MemoryPipeline.add_memory()`
   now does all three in one call, and only compares the *new* memory
   against existing ones (`discover_relationships_for()`) instead of
   rebuilding everything.

2. `backend/api/` was empty. `main.py` is a thin FastAPI wrapper over
   `MemoryPipeline` — it does not reimplement any logic, it just calls
   `add_memory()` and `remember()`.

## Install

```bash
cd MEMORA
pip install -r requirements-api.txt
```

(`uvicorn` was already in your venv; you're missing `fastapi` and
`python-multipart`, which is what the file above adds.)

## Run

From the project root (same place you already run `indexer.py` etc.
from, so `data/` paths and `backend.*` imports resolve the same way):

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

First startup will be slow — it loads the embedder, semantic
analyzer, reranker, and reconstruction model once, then keeps them
warm in memory for every request after that.

To let the iQOO phone (or Moiz's emulator) hit this from another
device on the same network, use your laptop's LAN IP instead of
`localhost`, or route it through Office Kit / `adb reverse` if you
want it phone-local.

## Endpoints

```
GET  /health
POST /ingest                (multipart file upload)
POST /remember               {"query": "..."}
GET  /memories
GET  /memory/{id}
GET  /memory/{id}/relationships
```

### Examples

```bash
# health check
curl http://localhost:8000/health

# ingest a new file — fully processed: OCR/parse -> semantic
# analysis -> chunk/embed/index -> relationship discovery
curl -F "file=@test_memory/PHOTO-2026-03-30-09-46-15.jpg" \
     http://localhost:8000/ingest

# ask a question
curl -X POST http://localhost:8000/remember \
     -H "Content-Type: application/json" \
     -d '{"query": "Find the handwritten assignment about defective bolts"}'

# list everything stored
curl http://localhost:8000/memories

# one memory's full detail
curl http://localhost:8000/memory/PHOTO-2026-03-30-09-46-15

# just its relationships
curl http://localhost:8000/memory/PHOTO-2026-03-30-09-46-15/relationships
```

`/remember` response shape (this is the contract Moiz should build
against):

```json
{
  "query": "...",
  "answer": "...",
  "confidence": 0.78,
  "evidence": ["test_memory\\PHOTO-....jpg", "..."],
  "memories": [
    {
      "memory_id": "PHOTO-2026-03-30-09-46-15",
      "source": "test_memory\\PHOTO-....jpg",
      "title": "...",
      "retrieval_type": "primary"
    }
  ]
}
```

## Known limitations, on purpose, given the time you have

- **No delete/dedupe endpoint.** `MemoryStore` itself has no
  `delete()` method (only relationship removal). Didn't add one —
  out of scope for a demo, and touching storage semantics this late
  is exactly the kind of thing your own notes warned against ("do
  not repeatedly rewrite `memory_store.py`").
- **`/ingest` is synchronous and blocking.** A big PDF or an image
  will make the HTTP request wait through OCR + semantic analysis +
  embedding before responding. Fine for a demo with a handful of
  files; would need a background job queue for anything larger.
- **CORS is wide open (`*`).** Deliberate for a same-weekend local
  demo. Don't ship this as-is beyond the hackathon.
- **Not tested against your actual models in this environment** — I
  don't have your Qwen/BLIP/Tesseract/Chroma setup here, so this is
  correct-by-inspection against your existing function signatures
  and verified for syntax, not run end-to-end. Run
  `curl -F "file=@..." http://localhost:8000/ingest` yourself first
  thing to catch anything environment-specific (e.g. your hardcoded
  Windows Tesseract path).

## If you still have time after this works

In priority order, matching what's weakest per your own progress
notes:

1. Rehearse the demo on 2–3 known-good files/queries. Judges score
   "does it work" (30%) and a 3–5 min pitch (10%) — a flaky live
   demo on an untested new file costs more than a polished one.
2. Make sure something phone-side actually calls this API during the
   demo (Office Kit screen-mirror hitting `/remember`, or a minimal
   screen in Moiz's app) — HackTracker's phone-use and Office Kit
   scoring is 25% combined, and a laptop-only demo scores zero there.
3. Only then: OCR normalization / semantic analyzer reliability —
   these are real weaknesses but open-ended, not "day-sized" fixes.