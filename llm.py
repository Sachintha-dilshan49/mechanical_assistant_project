"""
llm.py
One place to ask a language model for text, with automatic failover.

Providers are tried in order and the next one takes over when the current is
rate-limited or out of quota. Nothing else in the project needs to know which
one answered. Identical prompts are served from a local cache, which matters a
lot in practice: benchmarking and demos repeat the same queries, and every cache
hit is a query that costs no quota and returns instantly.

Two task profiles:
    "fast"  - query understanding, follow-up rewriting. Small, cheap models.
    "smart" - the reasoning step that picks and ranks materials.

Configure in .env:
    GROQ_API_KEY=...          # optional
    GEMINI_API_KEY=...        # optional
    DEEPSEEK_API_KEY=...      # optional; new accounts get 5M free tokens (30 days)
    LLM_CACHE=0               # optional; set to 0 to disable the prompt cache
    GROQ_MODEL_SMART=...      # optional overrides if a model id is retired
    GROQ_MODEL_FAST=...
    LLM_PRIMARY=groq|gemini        # optional; forces ONE provider for both tasks
    LLM_PRIMARY_FAST=gemini|groq   # optional; default gemini (fast + separate quota)
    LLM_PRIMARY_SMART=groq|gemini  # optional; default groq (larger token allowance)
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------
# MODEL LISTS
# Each provider tries its models in order. Model ids do get retired by
# providers; when one 404s we skip to the next automatically, and any id can be
# overridden from .env without touching this file. `py check_llm.py` lists the
# ids your key can actually reach.
# -----------------------------------------------------------------
GROQ_MODELS = {
    # gpt-oss-120b is the strongest reasoner this account can reach; the 20b is
    # the quick one for extraction work. Verified against the live model list
    # with check_llm.py - Groq retires ids, so re-run it if answers go empty.
    "smart": [os.getenv("GROQ_MODEL_SMART", "openai/gpt-oss-120b"),
              "qwen/qwen3.6-27b",
              "openai/gpt-oss-20b"],
    "fast":  [os.getenv("GROQ_MODEL_FAST", "openai/gpt-oss-20b"),
              "qwen/qwen3.6-27b",
              "openai/gpt-oss-120b"],
}

GEMINI_MODELS = {
    "smart": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
    "fast":  ["gemini-2.5-flash-lite", "gemini-2.5-flash"],
}

# DeepSeek is OpenAI-compatible at https://api.deepseek.com. There is no standing
# free tier - new accounts get a 5M-token grant that expires after 30 days - so it
# is a useful third fallback rather than something to lean on permanently.
DEEPSEEK_MODELS = {
    "smart": [os.getenv("DEEPSEEK_MODEL_SMART", "deepseek-v4-pro"), "deepseek-v4-flash"],
    "fast":  [os.getenv("DEEPSEEK_MODEL_FAST", "deepseek-v4-flash"), "deepseek-v4-pro"],
}

# Which provider leads for each task. The two tasks go to DIFFERENT providers on
# purpose: rate limits are per-model per-provider, so splitting them doubles the
# effective throughput instead of draining one token bucket twice per query.
# Measured: query understanding is ~2.7k tokens and gemini-2.5-flash-lite answers
# it in ~2s, while the bigger reasoning prompt is better served by Groq's larger
# free allowance. LLM_PRIMARY overrides both if you want one provider only.
TASK_PRIMARY = {
    "fast":  (os.getenv("LLM_PRIMARY_FAST") or "gemini").strip().lower(),
    "smart": (os.getenv("LLM_PRIMARY_SMART") or "groq").strip().lower(),
}

# How long to stop asking a provider after it reports an exhausted quota.
# Without this, every request would pay the failing call's latency before
# falling through to the other provider.
QUOTA_COOLDOWN_S = 15 * 60
TRANSIENT_RETRIES = 2

_cooldown_until = {}      # provider name -> unix time it becomes usable again
_clients = {}             # lazily built provider clients


# =================================================================
# CLIENTS  (built on first use so a missing key is never fatal at import)
# =================================================================
def _groq_client():
    if "groq" not in _clients:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            _clients["groq"] = None
        else:
            try:
                from groq import Groq
                _clients["groq"] = Groq(api_key=key)
            except ImportError:
                print("  [llm] groq package not installed - skipping Groq")
                _clients["groq"] = None
    return _clients["groq"]


def _gemini_client():
    if "gemini" not in _clients:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            _clients["gemini"] = None
        else:
            try:
                from google import genai
                _clients["gemini"] = genai.Client(api_key=key)
            except ImportError:
                print("  [llm] google-genai package not installed - skipping Gemini")
                _clients["gemini"] = None
    return _clients["gemini"]




def _deepseek_client():
    if "deepseek" not in _clients:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            _clients["deepseek"] = None
        else:
            try:
                from openai import OpenAI
                _clients["deepseek"] = OpenAI(api_key=key,
                                              base_url="https://api.deepseek.com")
            except ImportError:
                print("  [llm] openai package not installed - skipping DeepSeek")
                _clients["deepseek"] = None
    return _clients["deepseek"]


_CLIENT_GETTERS = {
    "groq": _groq_client,
    "gemini": _gemini_client,
    "deepseek": _deepseek_client,
}


def available_providers(task="smart"):
    """Provider names that have a usable key, in failover order for this task."""
    forced = (os.getenv("LLM_PRIMARY") or "").strip().lower()
    primary = forced or TASK_PRIMARY.get(task, "groq")
    # Primary first, then the rest as fallbacks. DeepSeek sits last by default:
    # its free grant expires, so it should absorb overflow rather than the load.
    order = [primary] + [p for p in ("groq", "gemini", "deepseek") if p != primary]
    return [p for p in order if _CLIENT_GETTERS.get(p, lambda: None)() is not None]


# =================================================================
# ERROR CLASSIFICATION
# Both SDKs raise their own exception types, so classify on status code and
# message text instead. That keeps this working across SDK versions.
# =================================================================
def _classify(exc):
    """One of: 'auth', 'quota', 'transient', 'missing_model', 'fatal'."""
    text = str(exc).lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)

    # A bad or expired key fails identically on every model, so treat it like a
    # dead provider rather than retrying the whole model list against it.
    if (status in (401, 403) or "invalid api key" in text
            or "unauthorized" in text or "api key not valid" in text
            or "permission denied" in text):
        return "auth"
    if status == 429 or "429" in text or "rate limit" in text \
            or "resource_exhausted" in text or "quota" in text \
            or "insufficient_quota" in text or "tokens per day" in text:
        return "quota"
    if status in (404, 400) and ("model" in text and
                                 ("not found" in text or "decommission" in text
                                  or "does not exist" in text or "deprecated" in text)):
        return "missing_model"
    if status in (500, 502, 503, 504) or "503" in text or "unavailable" in text \
            or "overloaded" in text or "timeout" in text or "connection" in text:
        return "transient"
    return "fatal"


def _on_cooldown(provider):
    return time.time() < _cooldown_until.get(provider, 0)


def _start_cooldown(provider, reason="quota exhausted"):
    _cooldown_until[provider] = time.time() + QUOTA_COOLDOWN_S
    print(f"  [llm] {provider} {reason} - pausing it for "
          f"{QUOTA_COOLDOWN_S // 60} min")


# =================================================================
# PER-PROVIDER CALLS
# =================================================================
# =================================================================
# USAGE LOG
# Providers do not offer a "how much have I spent today" endpoint, so the
# only reliable record is the one we keep. Every call reports its own token
# count; we store it and can answer the question offline, for free.
# =================================================================
def _record_usage(provider, model, prompt_tokens, completion_tokens):
    conn = _cache_conn()
    if conn is None:
        return
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS usage_log (
                            ts REAL NOT NULL,
                            provider TEXT NOT NULL,
                            model TEXT NOT NULL,
                            prompt_tokens INTEGER NOT NULL,
                            completion_tokens INTEGER NOT NULL)""")
        conn.execute("INSERT INTO usage_log VALUES (?, ?, ?, ?, ?)",
                     (time.time(), provider, model,
                      int(prompt_tokens or 0), int(completion_tokens or 0)))
        conn.commit()
        conn.close()
    except Exception:
        pass          # usage accounting must never break a request


def _usage_from(resp, provider, model):
    """Pull the token counts out of whichever SDK shape this is."""
    try:
        if provider == "gemini":
            u = getattr(resp, "usage_metadata", None)
            if u:
                _record_usage(provider, model,
                              getattr(u, "prompt_token_count", 0),
                              getattr(u, "candidates_token_count", 0))
        else:
            u = getattr(resp, "usage", None)
            if u:
                _record_usage(provider, model,
                              getattr(u, "prompt_tokens", 0),
                              getattr(u, "completion_tokens", 0))
    except Exception:
        pass


def usage_since(hours=24):
    """Tokens this app has spent per provider in the last N hours.

    Returns {provider: {"calls": n, "prompt": n, "completion": n, "total": n}}.
    This is what WE sent - the provider's own counter may differ slightly, and
    a rolling daily limit is measured on their clock, not ours.
    """
    conn = _cache_conn()
    if conn is None:
        return {}
    try:
        rows = conn.execute(
            """SELECT provider, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens)
               FROM usage_log WHERE ts > ? GROUP BY provider""",
            (time.time() - hours * 3600,)).fetchall()
        conn.close()
    except Exception:
        return {}
    out = {}
    for provider, calls, ptok, ctok in rows:
        ptok, ctok = ptok or 0, ctok or 0
        out[provider] = {"calls": calls, "prompt": ptok,
                         "completion": ctok, "total": ptok + ctok}
    return out


def quota_status():
    """What each provider says is left, where it will tell us.

    Groq returns x-ratelimit-* headers, so we read them from a 1-token call.
    Gemini does not expose remaining quota through the API at all. DeepSeek
    exposes an account balance rather than a rate limit.
    """
    status = []
    for provider in ("groq", "gemini", "deepseek"):
        if _CLIENT_GETTERS[provider]() is None:
            continue
        if provider == "groq":
            try:
                model = GROQ_MODELS["fast"][0]
                raw = _groq_client().chat.completions.with_raw_response.create(
                    model=model, messages=[{"role": "user", "content": "hi"}],
                    max_completion_tokens=1)
                h = raw.headers
                status.append({
                    "provider": "groq",
                    "tokens_remaining": h.get("x-ratelimit-remaining-tokens"),
                    "tokens_limit": h.get("x-ratelimit-limit-tokens"),
                    "requests_remaining": h.get("x-ratelimit-remaining-requests"),
                    "resets_in": h.get("x-ratelimit-reset-tokens"),
                    "note": "per-minute window; the daily cap shows up as a 429",
                })
            except Exception as exc:
                status.append({"provider": "groq", "error": str(exc)[:140]})
        elif provider == "deepseek":
            try:
                import httpx
                r = httpx.get("https://api.deepseek.com/user/balance",
                              headers={"Authorization":
                                       f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"},
                              timeout=15)
                info = r.json().get("balance_infos", [{}])[0]
                status.append({
                    "provider": "deepseek",
                    "balance": f"{info.get('total_balance')} {info.get('currency')}",
                    "note": "account balance, not a rate limit",
                })
            except Exception as exc:
                status.append({"provider": "deepseek", "error": str(exc)[:140]})
        else:
            status.append({
                "provider": "gemini",
                "note": "Gemini does not report remaining quota through the API - "
                        "check aistudio.google.com/app/apikey",
            })
    return status


def _call_groq(model, prompt, json_output):
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        # gpt-oss models spend completion tokens thinking before they emit the
        # answer; too small a cap truncates mid-JSON and the request 400s with
        # "max completion tokens reached before generating a valid document".
        "max_completion_tokens": 4096,
    }
    if json_output:
        # Guarantees syntactically valid JSON, which is stronger than asking for
        # it in the prompt. Not every model supports it - see the retry below.
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = _groq_client().chat.completions.create(**kwargs)
    except Exception as exc:
        if json_output and "response_format" in str(exc).lower():
            kwargs.pop("response_format")
            resp = _groq_client().chat.completions.create(**kwargs)
        else:
            raise
    _usage_from(resp, "groq", model)
    return resp.choices[0].message.content


def _call_gemini(model, prompt, json_output):
    config = {"max_output_tokens": 4096}
    if json_output:
        config["response_mime_type"] = "application/json"
    resp = _gemini_client().models.generate_content(
        model=model, contents=prompt, config=config
    )
    _usage_from(resp, "gemini", model)
    # .text is None when the model returns no text part (safety block, MAX_TOKENS,
    # or any non-STOP finish reason).
    return resp.text


def _call_deepseek(model, prompt, json_output):
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4096,
    }
    if json_output:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = _deepseek_client().chat.completions.create(**kwargs)
    except Exception as exc:
        if json_output and "response_format" in str(exc).lower():
            kwargs.pop("response_format")
            resp = _deepseek_client().chat.completions.create(**kwargs)
        else:
            raise
    _usage_from(resp, "deepseek", model)
    return resp.choices[0].message.content


_CALLERS = {"groq": _call_groq, "gemini": _call_gemini, "deepseek": _call_deepseek}
_MODELS = {"groq": GROQ_MODELS, "gemini": GEMINI_MODELS, "deepseek": DEEPSEEK_MODELS}


# =================================================================
# PROMPT CACHE
# The same prompt always gets the same answer back, without spending quota.
# This is not a micro-optimisation: a day of benchmarking and demoing repeats
# the same handful of queries, and that is exactly what exhausts a free tier.
# The cache is disposable - delete the file and it rebuilds.
# =================================================================
CACHE_FILE = "data/llm_cache.db"
CACHE_TTL_S = 30 * 24 * 3600
_cache_ready = False


def _cache_enabled():
    return (os.getenv("LLM_CACHE") or "1").strip() not in ("0", "false", "no")


def _cache_conn():
    """Cache connection, or None if the cache is off or unusable."""
    global _cache_ready
    if not _cache_enabled():
        return None
    try:
        import sqlite3
        conn = sqlite3.connect(CACHE_FILE)
        if not _cache_ready:
            conn.execute("""CREATE TABLE IF NOT EXISTS llm_cache (
                                key TEXT PRIMARY KEY,
                                response TEXT NOT NULL,
                                created_at REAL NOT NULL)""")
            conn.commit()
            _cache_ready = True
        return conn
    except Exception:
        return None       # a broken cache must never break generation


def _cache_key(prompt, task, json_output):
    import hashlib
    raw = f"{task}|{json_output}|{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key):
    conn = _cache_conn()
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT response, created_at FROM llm_cache WHERE key = ?", (key,)).fetchone()
        conn.close()
        if row and (time.time() - row[1]) < CACHE_TTL_S:
            return row[0]
    except Exception:
        pass
    return None


def _cache_put(key, response):
    conn = _cache_conn()
    if conn is None:
        return
    try:
        conn.execute("INSERT OR REPLACE INTO llm_cache (key, response, created_at) "
                     "VALUES (?, ?, ?)", (key, response, time.time()))
        conn.commit()
        conn.close()
    except Exception:
        pass


def cache_stats():
    """(entries, bytes) held in the prompt cache."""
    conn = _cache_conn()
    if conn is None:
        return (0, 0)
    try:
        n = conn.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(response)), 0) "
                         "FROM llm_cache").fetchone()
        conn.close()
        return (n[0], n[1])
    except Exception:
        return (0, 0)


# =================================================================
# PUBLIC API
# =================================================================
def generate(prompt, task="smart", json_output=False):
    """Generate text, failing over across models and providers.

    Returns (text, warning). `text` is None when every option failed; `warning`
    is a short note for the UI when the answer did not come from the primary
    provider, else None.
    """
    key = _cache_key(prompt, task, json_output)
    cached = _cache_get(key)
    if cached is not None:
        return cached, None

    providers = available_providers(task)
    if not providers:
        return None, ("No LLM API key configured - set GROQ_API_KEY, "
                      "GEMINI_API_KEY or DEEPSEEK_API_KEY in your .env file.")

    primary = providers[0]
    last_error = None

    for provider in providers:
        if _on_cooldown(provider):
            print(f"  [llm] skipping {provider} (cooling down after quota limit)")
            continue

        for model in _MODELS[provider][task]:
            for attempt in range(TRANSIENT_RETRIES):
                try:
                    text = _CALLERS[provider](model, prompt, json_output)
                    if not text:
                        break                      # empty part: try the next model
                    _cache_put(key, text)
                    warning = None
                    if provider != primary:
                        warning = (f"{primary.title()} was unavailable - answered "
                                   f"with {provider.title()} instead.")
                    return text, warning
                except Exception as exc:
                    last_error = exc
                    kind = _classify(exc)
                    if kind == "auth":
                        _start_cooldown(provider, "rejected the API key")
                        break                      # whole provider is out
                    if kind == "quota":
                        _start_cooldown(provider)
                        break                      # whole provider is out
                    if kind == "missing_model":
                        print(f"  [llm] {provider}/{model} unavailable - trying next model")
                        break                      # this model only
                    if kind == "transient":
                        wait = 2 * (attempt + 1)
                        print(f"  [llm] {provider}/{model} busy "
                              f"({attempt + 1}/{TRANSIENT_RETRIES}); waiting {wait}s")
                        time.sleep(wait)
                        continue
                    print(f"  [llm] {provider}/{model} error: {exc}")
                    break
            else:
                continue
            if _on_cooldown(provider):
                break                              # quota hit: skip its other models

    print(f"  [llm] all providers failed. Last error: {last_error}")
    return None, ("AI service unavailable on every configured provider - "
                  "showing database matches only.")


def strip_json_fences(text):
    """Remove ```json fences some models add despite being told not to."""
    if not text:
        return text
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```"))
    return text.strip()
