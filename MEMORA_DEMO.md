# MEMORA demo checklist

Run the memory pipeline from the repository root:

```powershell
.\venv\Scripts\python.exe -m backend.reconstruction.memory_pipeline
```

Use these queries to demonstrate the memory-side workflow:

1. `What were those probability questions I had about bolts?`
   - Expected: deterministic question recall from the assignment.
2. `What distributions were covered in my probability notes?`
   - Expected: stored distribution topics rather than an invented answer.
3. `How are the probability assignment and my probability notes related?`
   - Expected: a deterministic explanation from persisted relationship metadata, with shared topics/entities and confidence `0.85` for the current test memories.
4. `What do I remember about probability across my memories?`
   - Expected: deterministic multi-memory synthesis using the stored summaries and topics.
5. `What do I remember about quantum computing?`
   - Expected: `I don't have a stored memory that matches that.` with confidence `0.05`.

## Contextual relationships

When a memory has explicit `metadata.event_date` (or `date`/`occurred_at`) and structured `metadata.facts`, MEMORA stores temporal context and detects conflicting values for the same fact key. It reports the conflict and does not decide which memory is correct. File creation timestamps are intentionally not treated as event dates.

## Offline model setup

MEMORA loads models from the local Hugging Face cache only. Download required models once in a connected environment, then run the demo offline. This avoids slow network retries during normal local use.

## Deterministic synthesis benchmark

This benchmark does not load Qwen or the retrieval models. It measures the local multi-memory synthesis path at 10, 100, 500, and 1,000 memories:

```powershell
.\venv\Scripts\python.exe -m backend.reconstruction.benchmark_deterministic
```
