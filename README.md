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