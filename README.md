# Mechanical Engineering Material Selection Assistant

An AI tool that helps mechanical engineering students pick the right material for a design task. Describe the operating conditions in plain English; get back suitable metals with properties, reasoning, and citations.

**Status:** Work in progress. Full RAG pipeline working (query understanding → retrieval → reasoning → Streamlit UI) with 66 curated materials across 24 classes loaded. Every row is hand-verified with a cited source — bulk scraped datasets are deliberately excluded, since rows without corrosion, chemical-resistance and application context crowd out good candidates and degrade the recommendations.

---

## Why This Exists

Students waste hours flipping between Shigley, ASME standards, and datasheets for one design decision. Generic chatbots hallucinate values without citations. This tool runs RAG over a curated database — the LLM only reasons over verified data, never invents it.

> The LLM is the mouth. The database is the brain.

---

## Architecture

User query → LLM extracts filters → ChromaDB (semantic) + SQLite (exact values) → LLM reasons → Answer with citations


## Tech Stack

- **Python 3.10+**
- **ChromaDB** — vector database
- **sentence-transformers** — embeddings
- **SQLite** — built-in, for precise lookups
- **pandas + openpyxl** — read the materials spreadsheet
- **Streamlit** — UI
- **google-genai** — Gemini LLM for query understanding + reasoning

---

## Where the Data Lives

**SQLite (`data/materials.db`) is the source of truth.** It holds the materials
table and the allowable-stress tables. Missing properties are `NULL`.

**ChromaDB (`data/chroma_db/`) is a derived search index.** It holds embeddings
for semantic search, and a copy of each material's properties for filtering.
Delete it any time — `reindex.py` rebuilds it in under a minute. `NULL` becomes
the `NOT_FOUND` sentinel on the way in, since Chroma metadata can't hold nulls.

## Running

```bash
venv\Scripts\streamlit run app.py     # start the app
py init_db.py                         # first-time setup: seed SQLite from the CSVs
py reindex.py                         # rebuild the search index from SQLite
```

`init_db.py` refuses to overwrite a populated table (`--force` overrides), so
re-running it can't discard materials added after the initial seed.
`build_v15_dataset.py` regenerates the curated CSVs and is a one-time seed
source — it does not feed the live database.

---