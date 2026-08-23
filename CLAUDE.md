# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A RAG assistant that recommends engineering materials from a **curated, hand-verified** database
(66 materials / 24 classes). The guiding rule, from the README: *"The LLM is the mouth. The
database is the brain."* Every number in an answer must come from the database — the model selects,
ranks and explains, but never invents values or citations. Bulk-scraped rows are deliberately kept
out because rows lacking corrosion/chemical/application context crowd out good candidates.

## Commands

Windows, Python 3.11, `venv/` is committed-in-place (not in git) and scripts are run with `py` or
the venv interpreter:

```bash
venv\Scripts\streamlit run app.py   # run the UI
py init_db.py                       # seed SQLite from data/*.csv (refuses a non-empty table; --force wipes)
py reindex.py                       # rebuild data/chroma_db from SQLite (safe to run any time)
py check_llm.py                     # which providers/model ids the keys can reach + live call per task
py bench_accuracy.py                # accuracy + latency benchmark (10 cases, real LLM calls)
```

There is no pytest suite. "Tests" are:
- `py bench_accuracy.py` — the real regression gate: retrieval recall, top-1 correctness, any-in-top-3,
  and per-stage latency over 10 fixed queries with hand-picked acceptable `material_id` sets.
  **Run this after touching prompts, retrieval or pooling.** It makes live API calls and takes minutes.
- `py understand_query.py` / `py reason.py` — each module has a `__main__` demo block that runs a few
  queries end to end. Fastest way to check one stage in isolation.
- `py reindex.py` self-verifies the index (numeric filter, class filter, `NOT_FOUND` preservation).

Requirements file is `requirments.txt` (sic). Keys go in `.env`: `GROQ_API_KEY`, `GEMINI_API_KEY`.

## Architecture

Pipeline, all orchestrated by `reason_about_query()` in `reason.py`:

```
app.py (Streamlit chat)
  └─ reason.reason_about_query(query, top_k, history, on_step)
       ├─ small_talk_reply()        greetings/thanks/"what can you do" — answered with no API call
       ├─ contextualize_query()     rewrites a follow-up ("make it cheaper") into a standalone query
       ├─ understand_query()        LLM → {intent, semantic_query, filters, extracted_constraints}
       ├─ retrieve.find_candidates()  broad class-diverse pool (18) from ChromaDB
       ├─ LLM reasoning             SELECTS + RANKS from the pool, returns JSON
       └─ retrieve.get_allowable_stress()  exact SQLite lookup when a temperature was extracted
```

### Two invariants worth knowing before changing anything

**1. Filters gather; the LLM decides.** `find_candidates()` deliberately does *not* apply the numeric
filters as hard cutoffs to the final answer. A plain semantic search is dominated by whichever class
has the most rows and best keyword match (hundreds of aluminium alloys crowd steel out of a "car
body" query), so retrieval pulls the top 1–2 from *each* candidate class as guaranteed picks, then
fills by relevance. Tightening filters into a hard gate silently deletes the right answer before the
model ever sees it — that regression shows up in `bench_accuracy.py` as a recall miss.

**2. SQLite is the source of truth; ChromaDB is derived.** `data/materials.db` holds the `materials`
and `allowable_stress` tables (both gitignored, rebuilt from `data/*.csv`). `data/chroma_db/` is an
index you can delete at will — `reindex.py` rebuilds it in under a minute. NULL in SQL becomes the
string sentinel `"NOT_FOUND"` on the way into Chroma (Chroma metadata cannot hold nulls), and
`retrieve.py`, `reason.py` and `app.py` all read `NOT_FOUND` as "property does not apply". Numeric
fields stay floats so `$gte`/`$lte` keep working.

### Module map

- `db.py` — the **only** place that knows the schema (`MATERIAL_COLUMNS`, `KNOWN_CLASSES`,
  `RATING_COLUMNS`), the coercion helpers, and `to_chroma_metadata()`. Add a property here and the
  loader, reindexer and any future ingester stay in agreement. Ratings are CHECK-constrained to 1–5;
  `create_schema()` never drops anything.
- `llm.py` — single entry point `generate(prompt, task="fast"|"smart", json_output=)`, returning
  `(text, warning)`. Handles model-list walking, `_classify()` of errors (auth/quota/transient/
  missing_model/fatal), a 15-min per-provider cooldown after quota/auth failure, and provider
  failover. The two task profiles intentionally go to **different** providers by default (fast→Gemini,
  smart→Groq) because rate limits are per-provider — splitting them doubles effective throughput.
  Overridable via `LLM_PRIMARY`, `LLM_PRIMARY_FAST`, `LLM_PRIMARY_SMART`, `GROQ_MODEL_SMART`,
  `GROQ_MODEL_FAST`. Nothing else in the project should import a provider SDK directly.
- `understand_query.py` — one big `SYSTEM_PROMPT` (the field list, per-family filter rules, and
  worked examples) plus a thin JSON-parsing function. Adding a filterable column means editing both
  this prompt and `db.MATERIAL_COLUMNS`.
- `retrieve.py` — `find_materials()` (single Chroma query, auto-`$and`-wraps multi-condition filters),
  `find_candidates()` (the diversity pooling above), `infer_family_classes()` (keyword safety net so
  "plastic gear" stays in polymer classes even when the LLM emits no `material_class`), and
  `get_allowable_stress()` (SQLite, linear interpolation between table rows).
- `reason.py` — reasoning prompt, `format_material_for_prompt()` (emits only the fields that apply to
  that material's family, so the model can't read an absent field as zero), `coverage_note()` (built
  live from the class list; tells the model which families are *absent* so it says "you actually want
  rubber" instead of confidently recommending epoxy), and `build_fallback_summary()` for rule-based
  output when no LLM is reachable.
- `app.py` — Streamlit chat with multiple conversations in `st.session_state.conversations`, inline
  message editing, and `on_step` progress labels wired to `st.status`.

### Graceful degradation is a feature

Every stage degrades instead of raising: no LLM for understanding → raw text semantic search; no LLM
for reasoning → top pool matches with `build_fallback_summary()`; hallucinated `material_id`s in the
LLM's JSON are dropped because only ids present in `pool_by_id` are accepted. `result["warning"]`
carries the reason to the UI. Keep new code in this shape — the app must still answer with database
matches when both providers are down.

### Legacy / one-time scripts

`build_chromadb.py`, `build_sqlite.py`, `read_data.py` read the old Excel template and predate the
SQLite-as-source-of-truth design — do not use them; use `init_db.py` + `reindex.py`.
`build_v15_dataset.py` (150KB) regenerates the curated CSVs and is a one-time seed source that does
not feed the live database. `test_gemini.py` is a bare SDK smoke test superseded by `check_llm.py`.
