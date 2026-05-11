## Citation Granularity vs Chunk Count

- Attempted block-level PDF ingestion (per paragraph/heading/block) to enable
  exact line-level citations.
- A 70-page PDF that produces ~145 chunks at page-level ingestion explodes to ~1712 chunks at block-level an 11x increase making it impractical for larger documents. Currently using page-level chunking with first 80 character snippet citations as a compromise.
- True line-level citation remains an open problem.

## Retrieval Threshold Tuning

Initially set FAISS similarity threshold at `1.8` (strict). Queries like "submitted by?"
and "date of report?" were getting filtered out despite relevant chunks existing in the
index scores were landing just above the cutoff.

### Fix

Moved to a two-phase threshold system:

- **First pass:** `2.0` — primary retrieval with moderate strictness
- **Expansion retry:** `2.2` — lenient retrieval on LLM-expanded query

### Query Expansion Protocol

If first pass retrieval returns no documents, the system activates a query expansion
mechanism the LLM rewrites the original query with richer domain context using the
last 3 conversation exchanges as history, then retries retrieval at the lenient threshold.
Only if both passes fail does the system fall back to a general LLM response or "I don't know."

## FAISS Index Persistence

Previously the index was rebuilt from scratch on every run — re-embedding all chunks
regardless of whether the same PDF had been loaded before.

### Fix

FAISS index is now saved to disk after first build and reloaded on subsequent runs
for the same PDF(s). Index identity is determined by an MD5 hash of the sorted PDF
filenames — same files always map to the same index, different files build a new one.

### Structure

faiss_index/

- a3f1bc9d2e/ ← hash of "report final draft.pdf"
- 7c4e2a1f9b/ ← hash of "paper1.pdf"
- d8b3f2c1a4/ ← hash of "paper1.pdf + paper2.pdf"

### Behaviour

- **Same PDF(s):** loads index from disk instantly, skips embedding entirely
- **Different PDF(s):** detects hash mismatch, builds and saves a new index
- **Multiple PDFs:** treated as a combined session, order of selection does not matter
