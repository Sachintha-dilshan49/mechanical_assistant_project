# reason.py
# End-to-end RAG pipeline with graceful fallback when Gemini is unavailable.
# Returns structured data for the UI to render as cards.

import os
import json
import time
from dotenv import load_dotenv
from google import genai

from understand_query import understand_query
from retrieve import find_materials, get_allowable_stress

# -----------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# -----------------------------------------------------------------
# REASONING PROMPT (LLM only writes summary + per-material reasoning)
# -----------------------------------------------------------------
REASONING_PROMPT_TEMPLATE = """You are a senior mechanical engineer advising a student on material selection.

CRITICAL RULES:
1. Use ONLY the materials data provided below. No outside knowledge.
2. Every claim must come from the data. Cite the source.
3. If data is insufficient, say so.
4. For each material, write a 1-2 sentence reasoning ONLY (not full properties — the UI will show those).

The student asked: "{user_query}"

Our retrieval system understood this as:
- Semantic query: {semantic_query}
- Filters applied: {filters}

Materials retrieved (already ranked):
{materials_block}

{stress_block}

Output STRICT JSON in this exact format (no markdown fences, no other text):
{{
    "overall_summary": "<2-3 sentence comparison and recommendation>",
    "material_reasoning": {{
        "<material_id_1>": "<1-2 sentences explaining why this material fits the query>",
        "<material_id_2>": "<1-2 sentences>"
    }}
}}
"""


def format_material_for_prompt(m: dict) -> str:
    """Convert a retrieved material dict to a text block for the LLM."""
    return (
        f"--- {m.get('common_name')} (id: {m.get('material_id')}) ---\n"
        f"  yield_strength_MPa: {m.get('yield_strength_MPa')}\n"
        f"  max_service_temp_C: {m.get('max_service_temp_C')}\n"
        f"  density_kg_m3: {m.get('density_kg_m3')}\n"
        f"  corrosion_seawater: {m.get('corrosion_seawater')}/5\n"
        f"  weldability: {m.get('weldability')}/5\n"
        f"  machinability_index: {m.get('machinability_index')}\n"
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


def build_fallback_summary(materials: list, query: str) -> dict:
    """
    Build a basic summary from retrieval data alone, without LLM.
    Used when Gemini is completely unavailable.
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
        parts = []
        if mat.get("corrosion_seawater", 0) >= 4:
            parts.append("good seawater resistance")
        if mat.get("weldability", 0) >= 4:
            parts.append("easily weldable")
        if mat.get("machinability_index", 0) >= 70:
            parts.append("highly machinable")
        if mat.get("yield_strength_MPa", 0) >= 500:
            parts.append("high strength")
        if mat.get("cost_class") == 1:
            parts.append("low cost")
        
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
    
    # Step 2: Retrieve materials
    materials = find_materials(
        query_text=semantic_query,
        filters=filters if filters else None,
        top_k=top_k
    )
    if not materials:
        # Retry without filters if too restrictive
        materials = find_materials(query_text=semantic_query, top_k=top_k)
    if not materials:
        result["error"] = "No materials in the database match this query."
        return result
    result["materials"] = materials
    
    # Step 3: Stress lookups (database only, never fails on LLM)
    temp_C = constraints.get("temperature_C")
    if temp_C is not None:
        for mat in materials:
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
    
    # Step 4: Try LLM reasoning, fall back if Gemini fails
    materials_block = "\n\n".join(format_material_for_prompt(m) for m in materials)
    stress_block = ""
    if result["stress_lookups"]:
        stress_lines = [f"Stress at {temp_C}C:"]
        for mat_id, info in result["stress_lookups"].items():
            stress_lines.append(f"  - {mat_id}: {info['stress_MPa']} MPa")
        stress_block = "\n".join(stress_lines)
    
    final_prompt = REASONING_PROMPT_TEMPLATE.format(
        user_query=user_query,
        semantic_query=semantic_query,
        filters=json.dumps(filters),
        materials_block=materials_block,
        stress_block=stress_block,
    )
    
    try:
        text, fallback_note = call_gemini_with_fallback(final_prompt)
    except Exception as e:
        # Non-transient Gemini error (bad request, auth, network, etc.).
        # Never crash the pipeline — fall back to the rule-based summary.
        print(f"  Gemini reasoning failed: {e}")
        text, fallback_note = None, "AI summary unavailable - reasoning service errored. Showing database results only."

    if text:
        try:
            text = text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(line for line in lines if not line.startswith("```"))
            parsed = json.loads(text)
            result["summary"] = parsed.get("overall_summary", "")
            result["reasoning"] = parsed.get("material_reasoning", {})
            if fallback_note:
                result["warning"] = fallback_note
        except json.JSONDecodeError:
            # Malformed JSON from LLM - fall back to rule-based
            fallback = build_fallback_summary(materials, user_query)
            result["summary"] = fallback["overall_summary"]
            result["reasoning"] = fallback["material_reasoning"]
            result["warning"] = "AI returned malformed output - showing rule-based summary."
    else:
        # All Gemini attempts failed - use rule-based summary
        fallback = build_fallback_summary(materials, user_query)
        result["summary"] = fallback["overall_summary"]
        result["reasoning"] = fallback["material_reasoning"]
        result["warning"] = fallback_note or "AI summary unavailable. Showing database results only."
    
    return result


# -----------------------------------------------------------------
# DEMO (run from terminal)
# -----------------------------------------------------------------
if __name__ == "__main__":
    out = reason_about_query("lightweight material for marine use", top_k=3)
    print(json.dumps({k: v for k, v in out.items() if k != "materials"}, indent=2))
    print(f"\nMaterials returned: {len(out['materials'])}")