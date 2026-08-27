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

- **Python 3.11**
- **Streamlit** - chat UI
- **ChromaDB** + **sentence-transformers** - semantic search over material descriptions
- **SQLite** - the materials record store and the allowable-stress tables
- **groq / google-genai / openai** - three LLM providers behind one interface (`llm.py`)

---

## File Structure

```
mechanical_assistant_project/
|
|-- app.py                  Streamlit UI - chat, material cards, comparison table
|
|-- THE PIPELINE  (a query flows down through these, in order)
|   |-- reason.py           Orchestrator: intent gate -> retrieve -> rank -> stress lookup
|   |-- understand_query.py Plain English -> structured filters + intent
|   |-- retrieve.py         ChromaDB semantic search + SQLite stress lookups
|   |-- llm.py              All LLM calls: provider failover, quota cooldowns, cache
|
|-- THE DATA LAYER
|   |-- db.py               The 43-column schema. Every DB read/write goes through here
|   |-- init_db.py          One-time: seed SQLite from the curated CSVs
|   |-- reindex.py          Rebuild the ChromaDB index from SQLite (run after edits)
|   |-- build_v15_dataset.py  Regenerates the curated CSVs (seed source, not live data)
|
|-- TOOLS
|   |-- check_llm.py        Which providers/models your keys reach + cache status
|   |-- bench_accuracy.py   Retrieval recall and top-1 accuracy over test queries
|
|-- data/
|   |-- materials.db        SOURCE OF TRUTH - materials + allowable_stress
|   |-- chroma_db/          Derived search index (safe to delete, rebuild with reindex.py)
|   |-- llm_cache.db        Cached LLM responses (safe to delete)
|   |-- materials.csv       Curated seed data
|   |-- stress_temperature.csv
|
|-- archive/                Superseded scripts, kept for reference. Do not run.
|-- .env                    API keys (never commit)
|-- requirments.txt         Dependencies
```

### How a query flows

```
      "a boat fitting that won't corrode"
                  |
      app.py -----+
                  v
      reason.py  --> is this even a material question?      (no API call)
                  |     greeting / about the tool / off-topic -> answer directly
                  v
      understand_query.py --> {filters, constraints, intent}   [LLM: fast]
                  v
      retrieve.py  --> 18 class-diverse candidates from ChromaDB
                  v
      reason.py  --> LLM selects and ranks from that pool      [LLM: smart]
                  v
      retrieve.py  --> allowable stress at temperature, from SQLite
                  v
      app.py  --> ranked cards with properties, warnings, sources
```

The LLM never supplies a property value. It chooses among rows and explains the
choice; every number on screen comes from SQLite.

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