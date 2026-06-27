# reason.py
# End-to-end RAG pipeline with graceful fallback when Gemini is unavailable.
# Returns structured data for the UI to render as cards.

import os
import json
import time
from dotenv import load_dotenv
from google import genai

from understand_query import understand_query
from retrieve import find_candidates, get_allowable_stress

# -----------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# -----------------------------------------------------------------
# REASONING PROMPT (LLM only writes summary + per-material reasoning)
# -----------------------------------------------------------------
REASONING_PROMPT_TEMPLATE = """You are a senior mechanical engineer helping a student choose a material.
You are given a POOL of candidate materials retrieved from a database. Your job is to SELECT and
RANK the ones that genuinely fit the application — NOT to repeat the retrieval order. The pool was
gathered broadly and WILL contain unsuitable options; filtering them out by reasoning is your job.

The student asked: "{user_query}"
Interpreted as: {semantic_query}

CRITICAL RULES:
1. All specific property VALUES and numbers come ONLY from the data below — never invent a number
   and never cite a source you were not given. The database is the source of truth for facts.
2. Apply real engineering judgment and physical common sense to pick the best materials for THIS
   job — e.g. low density floats while dense ferrous metals sink and rust in water; brittle
   materials shatter under impact; high hardness resists wear; car bodies are normally formable
   sheet steel or aluminum; cheap toys are usually plastic. Reasoning from principles is required;
   fabricating data is not.
3. SELECT the best {max_select} or fewer materials, ranked best-first, and ignore poor fits. If NONE
   of the pool is well suited, still pick the closest 1-2, say clearly they are compromises, and
   describe what property profile would actually suit the job.
4. NEVER invent a value for a field shown as NOT_FOUND; if a property does not apply to the
   material's class, treat it as "not applicable".

INTERPRET PROPERTIES BY FAMILY (material_class):
- METALS: corrosion ratings, weldability, machinability.
- PLASTICS (plastic_*): hardness_shore_d (NOT hardness_HB), max_continuous_use_temp_C (NOT
  max_service_temp_C), chemical_resistance_* (NOT corrosion_*); mention flammability/uv_resistance/
  water_absorption when relevant.
- CERAMICS (ceramic_*): no yield point (NOT_FOUND) — UTS is the flexural strength; ALWAYS warn it is
  brittle and must be designed for compression.
- COMPOSITES (composite_*): properties are directional (especially composite_cfrp); for cfrp note
  that galvanic isolation from aluminium is required.

CANDIDATE POOL:
{materials_block}

{stress_block}

Output STRICT JSON only (no markdown fences, no other text):
{{
    "overall_summary": "<2-3 sentences: the recommendation and the key tradeoff>",
    "selected": [
        {{"material_id": "<id from the pool above>", "reasoning": "<1-2 sentences why it fits this job>"}}
    ]
}}
The "selected" list must be ranked best-first and contain ONLY material_id values that appear in the
pool above. Order matters — the first entry is your #1 recommendation."""


def format_material_for_prompt(m: dict) -> str:
    """Convert a retrieved material dict to a text block for the LLM.

    Includes both metal and non-metal property lines; fields that don't apply to
    this material show as NOT_FOUND so the LLM knows not to cite them. The prompt
    rules tell it which fields to use for each material family."""
    return (
        f"--- {m.get('common_name')} (id: {m.get('material_id')}) ---\n"
        f"  material_class: {m.get('material_class')}\n"
        f"  yield_strength_MPa: {m.get('yield_strength_MPa')}\n"
        f"  ultimate_tensile_strength_MPa: {m.get('ultimate_tensile_strength_MPa')}  (flexural strength for ceramics)\n"
        f"  hardness_HB: {m.get('hardness_HB')}   hardness_shore_d: {m.get('hardness_shore_d')}\n"
        f"  density_kg_m3: {m.get('density_kg_m3')}   elastic_modulus_GPa: {m.get('elastic_modulus_GPa')}\n"
        f"  max_service_temp_C: {m.get('max_service_temp_C')}   max_continuous_use_temp_C: {m.get('max_continuous_use_temp_C')}\n"
        f"  corrosion_seawater: {m.get('corrosion_seawater')}/5  (metals)\n"
        f"  chemical_resistance solvents/acids/alkalis/fuels: "
        f"{m.get('chemical_resistance_solvents')}/{m.get('chemical_resistance_acids')}/"
        f"{m.get('chemical_resistance_alkalis')}/{m.get('chemical_resistance_fuels')}  (non-metals, 1-5)\n"
        f"  weldability: {m.get('weldability')}/5   joining_method: {m.get('joining_method')}\n"
        f"  machinability_index: {m.get('machinability_index')}   "
        f"cost_class: {m.get('cost_class')}/5   approx_cost_usd_per_kg: {m.get('approx_cost_usd_per_kg')}\n"
        f"  flammability(UL94): {m.get('flammability')}   uv_resistance: {m.get('uv_resistance')}/5   "
        f"water_absorption_percent: {m.get('water_absorption_percent')}\n"
        f"  key_warnings: {m.get('key_warnings', '')}\n"
        f"  sources: {m.get('sources', '')}\n"
    )


# -----------------------------------------------------------------
# THREE-LAYER FALLBACK FOR GEMINI
# -----------------------------------------------------------------
def call_gemini_with_fallback(prompt: str):
    """
    Try gemini-2.5-flash first, fall back to gemini-2.5-flash-lite if it fails.
    Returns (text, warning_message). text is None if all attempts failed.
    """
    models_to_try = [
        ("gemini-2.5-flash", None),
        ("gemini-2.5-flash-lite", "Used the lite model (main model busy) - quality may be slightly lower."),
    ]
    
    for model_name, fallback_note in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text, fallback_note
            except Exception as e:
                error_str = str(e)
                if "503" in error_str or "429" in error_str or "UNAVAILABLE" in error_str:
                    wait = 3 * (attempt + 1)
                    print(f"  {model_name} busy (attempt {attempt + 1}/2). Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        print(f"  {model_name} unavailable. Trying next model...")
    
    return None, "AI summary unavailable - Gemini is overloaded. Showing database results only."


def _as_num(val):
    """Coerce a metadata value to float, or None if missing / NOT_FOUND.
    Lets the rule-based fallback compare values without crashing on the
    'NOT_FOUND' sentinel that non-applicable fields now carry."""
    try:
        if val is None:
            return None
        s = str(val).strip()
        if s == "" or s.upper() == "NOT_FOUND":
            return None
        return float(s)
    except (TypeError, ValueError):
        return None


def build_fallback_summary(materials: list, query: str) -> dict:
    """
    Build a basic summary from retrieval data alone, without LLM.
    Used when Gemini is completely unavailable. Handles metals and non-metals,
    and is safe against NOT_FOUND values.
    """
    if not materials:
        return {"overall_summary": "", "material_reasoning": {}}

    top = materials[0]
    summary = (
        f"Found {len(materials)} candidate materials matching your query. "
        f"The top match is {top.get('common_name')}. "
        f"Database results are shown below - review the property cards "
        f"to compare materials. (AI-written summary unavailable due to service load.)"
    )

    reasoning = {}
    for mat in materials:
        cls = str(mat.get("material_class", ""))
        parts = []

        # metal-oriented signals
        if (_as_num(mat.get("corrosion_seawater")) or 0) >= 4:
            parts.append("good seawater resistance")
        if (_as_num(mat.get("weldability")) or 0) >= 4:
            parts.append("easily weldable")
        if (_as_num(mat.get("machinability_index")) or 0) >= 70:
            parts.append("highly machinable")

        # non-metal signals
        if (_as_num(mat.get("chemical_resistance_acids")) or 0) >= 4 \
                or (_as_num(mat.get("chemical_resistance_solvents")) or 0) >= 4:
            parts.append("strong chemical resistance")
        shore = _as_num(mat.get("hardness_shore_d"))
        if shore is not None and shore >= 75:
            parts.append("hard, wear-resistant surface")

        # shared strength / cost signals (UTS covers ceramics & composites)
        if (_as_num(mat.get("yield_strength_MPa")) or 0) >= 500 \
                or (_as_num(mat.get("ultimate_tensile_strength_MPa")) or 0) >= 800:
            parts.append("high strength")
        if (_as_num(mat.get("cost_class")) or 99) <= 1:
            parts.append("low cost")

        # family-specific safety notes
        if cls.startswith("ceramic_"):
            parts.append("brittle — design for compression")
        if cls == "composite_cfrp":
            parts.append("directional properties")

        if parts:
            reasoning[mat["material_id"]] = (
                f"Matched the query with: {', '.join(parts)}. "
                f"See typical applications and warnings below for details."
            )
        else:
            reasoning[mat["material_id"]] = (
                "Retrieved as a candidate by semantic relevance. "
                "Review properties and warnings below."
            )

    return {"overall_summary": summary, "material_reasoning": reasoning}


# -----------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------
def reason_about_query(user_query: str, top_k: int = 3) -> dict:
    """
    Full RAG pipeline with graceful degradation.
    Always returns materials if any matched; LLM summary is best-effort.
    """
    result = {
        "query": user_query,
        "understood": None,
        "materials": [],
        "stress_lookups": {},
        "summary": "",
        "reasoning": {},
        "warning": None,
        "error": None,
    }
    
    # Step 1: Understand query (with fallback)
    understood = None
    try:
        understood = understand_query(user_query)
    except Exception as e:
        print(f"  understand_query failed: {e}")
    
    if not understood:
        understood = {
            "semantic_query": user_query,
            "filters": {},
            "extracted_constraints": {},
            "reasoning": "Query understanding unavailable; using raw text for semantic search.",
        }
        result["warning"] = "Could not parse query structure - searched semantically only."
    
    result["understood"] = understood
    
    semantic_query = understood.get("semantic_query", user_query)
    filters = understood.get("filters", {})
    constraints = understood.get("extracted_constraints", {})
    
    family = constraints.get("material_family")

    # Step 2: Retrieve a BROAD candidate pool. Filters only gather candidates here;
    # the reasoning LLM makes the actual choice, so a too-tight filter can no longer
    # silently delete the right answer before the model ever sees it.
    POOL_SIZE = 15
    pool = find_candidates(
        query_text=semantic_query,
        filters=filters if filters else None,
        family=family,
        pool_size=POOL_SIZE,
    )
    if not pool:
        result["error"] = "No materials in the database match this query."
        return result
    pool_by_id = {m["material_id"]: m for m in pool}

    # Step 3: Ask the LLM to SELECT and RANK the best materials from the pool.
    materials_block = "\n\n".join(format_material_for_prompt(m) for m in pool)
    final_prompt = REASONING_PROMPT_TEMPLATE.format(
        user_query=user_query,
        semantic_query=semantic_query,
        max_select=top_k,
        materials_block=materials_block,
        stress_block="",
    )

    try:
        text, fallback_note = call_gemini_with_fallback(final_prompt)
    except Exception as e:
        # Non-transient Gemini error — never crash the pipeline; fall back below.
        print(f"  Gemini reasoning failed: {e}")
        text, fallback_note = None, "AI selection unavailable - reasoning service errored. Showing top database matches."

    selected = []
    reasoning = {}
    summary = ""

    if text:
        try:
            text = text.strip()
            if text.startswith("```"):
                text = "\n".join(l for l in text.split("\n") if not l.startswith("```"))
            parsed = json.loads(text)
            summary = parsed.get("overall_summary", "")
            for item in parsed.get("selected", []):
                mid = item.get("material_id")
                if mid in pool_by_id and mid not in reasoning:   # ignore hallucinated/dupe ids
                    selected.append(pool_by_id[mid])
                    reasoning[mid] = item.get("reasoning", "")
                if len(selected) >= top_k:
                    break
            if fallback_note:
                result["warning"] = fallback_note
        except json.JSONDecodeError:
            result["warning"] = "AI returned malformed output - showing top database matches."

    if not selected:
        # Fallback: take the best pool matches with rule-based reasoning.
        selected = pool[:top_k]
        fb = build_fallback_summary(selected, user_query)
        summary = summary or fb["overall_summary"]
        reasoning = fb["material_reasoning"]
        if not result["warning"]:
            result["warning"] = fallback_note or "AI summary unavailable. Showing top database matches."

    result["materials"] = selected
    result["summary"] = summary
    result["reasoning"] = reasoning

    # Step 4: Allowable-stress lookups for the SELECTED materials (database only).
    temp_C = constraints.get("temperature_C")
    if temp_C is not None:
        for mat in selected:
            stress_table_id = mat.get("stress_table_id", "")
            if stress_table_id:
                stress_info = get_allowable_stress(stress_table_id, float(temp_C))
                if stress_info and stress_info.get("stress_MPa") is not None:
                    result["stress_lookups"][mat["material_id"]] = {
                        "stress_MPa": stress_info["stress_MPa"],
                        "source": stress_info["source"],
                        "interpolated": stress_info.get("interpolated", False),
                        "temperature_C": temp_C,
                    }

    return result


# -----------------------------------------------------------------
# DEMO (run from terminal)
# -----------------------------------------------------------------
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    for q in [
        "lightweight material for marine use",
        "low-friction plastic gear that resists fuels",
        "brittle electrical insulator for 1000 C",
    ]:
        print("\n" + "=" * 70)
        print(f"QUERY: {q}")
        print("=" * 70)
        out = reason_about_query(q, top_k=3)
        print(json.dumps({k: v for k, v in out.items() if k != "materials"}, indent=2))
        print(f"Materials returned: {len(out['materials'])} -> "
              f"{[m.get('common_name') for m in out['materials']]}")