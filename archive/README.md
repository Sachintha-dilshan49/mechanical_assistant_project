# Archived scripts

Superseded code, kept for reference. Nothing here is imported by the
running app, and nothing here should be run - several of these still
read the old Excel spreadsheet and would write stale data.

**build_chromadb.py** - Built ChromaDB straight from the Excel sheet. Replaced by init_db.py (CSV -> SQLite) + reindex.py (SQLite -> ChromaDB), because SQLite is now the source of truth and Chroma is rebuilt from it.

**build_sqlite.py** - Loaded only the stress tables from the Excel sheet. init_db.py does this now, together with the materials table.

**read_data.py** - One-off helper to print the Excel sheet while designing the schema. The schema now lives in db.py.

**test_gemini.py** - One-off Gemini connectivity check from when Gemini was the only provider. check_llm.py replaces it and covers all three providers, model availability and the prompt cache.
