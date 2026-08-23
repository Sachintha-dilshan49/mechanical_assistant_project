"""
check_llm.py
Shows which LLM providers your keys can reach, which model ids actually work,
and proves the Groq -> Gemini failover chain is wired up.

Run this after adding a key, or when answers suddenly come back empty:
    py check_llm.py
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import os
import llm

print("=" * 62)
print("LLM provider check")
print("=" * 62)

# -----------------------------------------------------------------
# Keys
# -----------------------------------------------------------------
print("\n[1] API keys in .env")
for name in ("GROQ_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY"):
    key = os.getenv(name)
    state = f"set ({len(key)} chars)" if key else "MISSING"
    print(f"    {name:<16} {state}")

providers = llm.available_providers("smart")
if not providers:
    print("\n    No usable provider. Add a key to .env and re-run.")
    raise SystemExit(1)
print(f"\n    fast  task order: {' -> '.join(llm.available_providers('fast'))}")
print(f"    smart task order: {' -> '.join(llm.available_providers('smart'))}")

entries, size = llm.cache_stats()
print(f"\n    Prompt cache: {entries} cached responses ({size} chars). "
      f"A cache hit costs no quota and returns instantly.")

if "deepseek" in providers:
    print("\n    NOTE: DeepSeek's free grant is 5M tokens and expires 30 days after")
    print("          signup - treat it as overflow capacity, not a permanent tier.")

# -----------------------------------------------------------------
# What models the Groq key can actually see
# -----------------------------------------------------------------
if "groq" in providers:
    print("\n[2] Model ids your Groq key can reach")
    try:
        models = sorted(m.id for m in llm._groq_client().models.list().data)
        for m in models:
            configured = any(m in v for v in llm.GROQ_MODELS.values())
            print(f"    {'* ' if configured else '  '}{m}")
        print("\n    (* = one this project is configured to use)")

        for task, wanted in llm.GROQ_MODELS.items():
            missing = [w for w in wanted if w not in models]
            if missing:
                print(f"    WARNING: '{task}' lists unavailable ids: {missing}")
                print(f"             Set GROQ_MODEL_{task.upper()} in .env to one above.")
    except Exception as exc:
        print(f"    Could not list models: {exc}")

# -----------------------------------------------------------------
# Live calls
# -----------------------------------------------------------------
if "deepseek" in providers:
    print("\n[2b] Model ids your DeepSeek key can reach")
    try:
        ds = sorted(m.id for m in llm._deepseek_client().models.list().data)
        for m in ds:
            configured = any(m in v for v in llm.DEEPSEEK_MODELS.values())
            print(f"    {'* ' if configured else '  '}{m}")
        for task, wanted in llm.DEEPSEEK_MODELS.items():
            missing = [w for w in wanted if w not in ds]
            if missing:
                print(f"    WARNING: '{task}' lists unavailable ids: {missing}")
                print(f"             Set DEEPSEEK_MODEL_{task.upper()} in .env.")
    except Exception as exc:
        print(f"    Could not list models: {exc}")

print("\n[3] Live call per task profile")
for task in ("fast", "smart"):
    text, warning = llm.generate(
        "Reply with exactly one word: OK", task=task
    )
    got = (text or "").strip().replace("\n", " ")[:40]
    print(f"    {task:<6} -> {got!r}" + (f"   [{warning}]" if warning else ""))

print("\n[4] JSON mode (used by query understanding and reasoning)")
text, warning = llm.generate(
    'Return JSON only: {"status": "ok"}', task="fast", json_output=True
)
print(f"    raw: {(text or '').strip()[:60]!r}")

print("\n" + "=" * 62)
print("Check complete")
print("=" * 62)
