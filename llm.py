"""
llm.py
One place to ask a language model for text, with automatic failover.

Groq is tried first (its free tier allows far more tokens per day than Gemini's),
and Gemini takes over when Groq is rate-limited or out of quota. Nothing else in
the project needs to know which provider answered.

Two task profiles:
    "fast"  - query understanding, follow-up rewriting. Small, cheap models.
    "smart" - the reasoning step that picks and ranks materials.

Configure in .env:
    GROQ_API_KEY=...          # optional; without it Gemini is used alone
    GEMINI_API_KEY=...        # optional; without it Groq is used alone
    GROQ_MODEL_SMART=...      # optional overrides if a model id is retired
    GROQ_MODEL_FAST=...
    LLM_PRIMARY=groq|gemini   # optional; defaults to groq
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


def available_providers():
    """Provider names that have a usable key, in failover order."""
    primary = (os.getenv("LLM_PRIMARY") or "groq").strip().lower()
    order = ["groq", "gemini"] if primary != "gemini" else ["gemini", "groq"]
    return [p for p in order
            if (_groq_client() if p == "groq" else _gemini_client()) is not None]


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
    if status in (401, 403) or "invalid api key" in text or "unauthorized" in text             or "api key not valid" in text or "permission denied" in text:
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
def _call_groq(model, prompt, json_output):
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
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
    return resp.choices[0].message.content


def _call_gemini(model, prompt, json_output):
    config = {"response_mime_type": "application/json"} if json_output else None
    resp = _gemini_client().models.generate_content(
        model=model, contents=prompt, config=config
    )
    # .text is None when the model returns no text part (safety block, MAX_TOKENS,
    # or any non-STOP finish reason).
    return resp.text


_CALLERS = {"groq": _call_groq, "gemini": _call_gemini}
_MODELS = {"groq": GROQ_MODELS, "gemini": GEMINI_MODELS}


# =================================================================
# PUBLIC API
# =================================================================
def generate(prompt, task="smart", json_output=False):
    """Generate text, failing over across models and providers.

    Returns (text, warning). `text` is None when every option failed; `warning`
    is a short note for the UI when the answer did not come from the primary
    provider, else None.
    """
    providers = available_providers()
    if not providers:
        return None, ("No LLM API key configured - set GROQ_API_KEY or "
                      "GEMINI_API_KEY in your .env file.")

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
