# reason.py
# The end-to-end pipeline: query → understanding → retrieval → reasoning
# This is the complete RAG system.

import os
import json
from dotenv import load_dotenv
from google import genai

# Import functions from our existing scripts
from understand_query import understand_query
from retrieve import find_materials, get_allowable_stress

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# -----------------------------------------------------------------
# THE REASONING PROMPT — strict grounding rules
# -----------------------------------------------------------------
REASONING_PROMPT_TEMPLATE = """You are a senior mechanical engineer advising a student on material selection.

CRITICAL RULES — these are non-negotiable:
1. You MUST use ONLY the materials data provided below. Do NOT use any knowledge outside this data.
2. Every property value you mention MUST come from the data block, never invented.
3. Every claim MUST include a citation from the "sources" field of the material.
4. If the data is insufficient to answer something, SAY SO. Do not guess.
5. Always include the "key_warnings" for any recommended material.
6. If the student gave a temperature and we looked up exact stress, mention it explicitly.

The student asked: "{user_query}"

Our retrieval system understood this as:
- Semantic query: {semantic_query}
- Filters applied: {filters}
- Constraints extracted: {constraints}

Materials retrieved from database (ranked by relevance):

{materials_block}

{stress_block}

Now write a concise engineering recommendation in this format:

## Recommended Materials

### 1. [Material Name]
- **Why this fits:** [1-2 sentences explaining match to the query]
- **Key properties:** [yield strength, max temp, corrosion ratings — only what's relevant]
- **Watch out for:** [the key_warnings from the data]
- **Sources:** [list the citations]

### 2. [Material Name] (if applicable)
[same format]

## Summary
[2-3 sentences comparing the options and suggesting which is best for the stated use case]

Be concise. No fluff. Engineering-grade communication.
"""


def format_material_for_prompt(material_dict: dict) -> str:
    """
    Convert a material dict from retrieve.py into a detailed text block for the LLM.
    Includes all properties the LLM needs to reason and warn correctly.
    """
    lines = [
        f"--- {material_dict.get('common_name', 'Unknown')} ---",
        f"  material_id: {material_dict.get('material_id', '')}",
        f"  material_class: {material_dict.get('material_class', '')}",
        f"  condition: {material_dict.get('condition', '')}",
        f"  yield_strength_MPa: {material_dict.get('yield_strength_MPa', 'N/A')}",
        f"  ultimate_tensile_strength_MPa: {material_dict.get('ultimate_tensile_strength_MPa', 'N/A')}",
        f"  density_kg_m3: {material_dict.get('density_kg_m3', 'N/A')}",
        f"  elastic_modulus_GPa: {material_dict.get('elastic_modulus_GPa', 'N/A')}",
        f"  max_service_temp_C: {material_dict.get('max_service_temp_C', 'N/A')}",
        f"  min_service_temp_C: {material_dict.get('min_service_temp_C', 'N/A')}",
        f"  corrosion_seawater: {material_dict.get('corrosion_seawater', 'N/A')}/5",
        f"  corrosion_acidic: {material_dict.get('corrosion_acidic', 'N/A')}/5",
        f"  corrosion_alkaline: {material_dict.get('corrosion_alkaline', 'N/A')}/5",
        f"  corrosion_atmospheric: {material_dict.get('corrosion_atmospheric', 'N/A')}/5",
        f"  corrosion_high_temp: {material_dict.get('corrosion_high_temp', 'N/A')}/5",
        f"  weldability: {material_dict.get('weldability', 'N/A')}/5",
        f"  weldability_notes: {material_dict.get('weldability_notes', '')}",
        f"  machinability_index: {material_dict.get('machinability_index', 'N/A')}",
        f"  cost_class: {material_dict.get('cost_class', 'N/A')}/5",
        f"  approx_cost_usd_per_kg: {material_dict.get('approx_cost_usd_per_kg', '')}",
        f"  availability: {material_dict.get('availability', 'N/A')}/5",
        f"  fatigue_rating: {material_dict.get('fatigue_rating', 'N/A')}/5",
        f"  typical_applications: {material_dict.get('typical_applications', '')}",
        f"  key_warnings: {material_dict.get('key_warnings', '')}",
        f"  sources: {material_dict.get('sources', '')}",
        f"  relevance_score: {material_dict.get('relevance_score', 0):.2f}",
    ]
    return "\n".join(lines)


def reason_about_query(user_query: str, top_k: int = 3) -> str:
    """
    The full RAG pipeline. Takes a user query, returns a final answer string.
    """
    print("\n" + "=" * 70)
    print(f"USER QUERY: {user_query}")
    print("=" * 70)
    
    # ----- Step 1: Understand the query -----
    print("\n[1/4] Understanding query with Gemini...")
    understood = understand_query(user_query)
    if not understood:
        return "ERROR: Could not understand the query. Try rephrasing."
    
    semantic_query = understood.get("semantic_query", user_query)
    filters = understood.get("filters", {})
    constraints = understood.get("extracted_constraints", {})
    
    print(f"      Semantic query: {semantic_query}")
    print(f"      Filters: {json.dumps(filters)}")
    print(f"      Constraints: {constraints}")
    
    # ----- Step 2: Retrieve materials -----
    print(f"\n[2/4] Searching database (top {top_k} results)...")
    materials = find_materials(
        query_text=semantic_query,
        filters=filters if filters else None,
        top_k=top_k
    )
    
    if not materials:
        # Retry without filters in case they were too restrictive
        print("      No matches with filters — retrying without filters")
        materials = find_materials(query_text=semantic_query, top_k=top_k)
    
    if not materials:
        return "No materials in the database match this query. Try broadening your criteria."
    
    print(f"      Found {len(materials)} candidate materials")
    
    # ----- Step 3: Look up exact stress if temperature was given -----
    stress_block = ""
    temp_C = constraints.get("temperature_C")
    if temp_C is not None:
        print(f"\n[3/4] Looking up allowable stress at {temp_C}°C for each candidate...")
        stress_lines = ["Exact allowable stress lookups (from SQLite/ASME tables):"]
        for mat in materials:
            stress_table_id = mat.get("stress_table_id", "")
            if stress_table_id:
                stress_info = get_allowable_stress(stress_table_id, float(temp_C))
                if stress_info and stress_info.get("stress_MPa") is not None:
                    interp_note = " (interpolated)" if stress_info.get("interpolated") else " (exact)"
                    stress_lines.append(
                        f"  - {mat['common_name']}: {stress_info['stress_MPa']} MPa "
                        f"at {temp_C}°C{interp_note}, source: {stress_info['source']}"
                    )
                elif stress_info and stress_info.get("error"):
                    stress_lines.append(f"  - {mat['common_name']}: {stress_info['error']}")
            else:
                stress_lines.append(f"  - {mat['common_name']}: no stress table available")
        stress_block = "\n".join(stress_lines)
    else:
        print(f"\n[3/4] No temperature specified — skipping stress lookup")
    
    # ----- Step 4: Reasoning step — Gemini writes the answer -----
    print(f"\n[4/4] Asking Gemini to write the recommendation...")
    
    materials_block = "\n\n".join(format_material_for_prompt(m) for m in materials)
    
    final_prompt = REASONING_PROMPT_TEMPLATE.format(
        user_query=user_query,
        semantic_query=semantic_query,
        filters=json.dumps(filters),
        constraints=constraints,
        materials_block=materials_block,
        stress_block=stress_block
    )
    
    # Retry logic for transient Gemini errors (503, 429, etc.)
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=final_prompt
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if "503" in error_str or "429" in error_str or "UNAVAILABLE" in error_str:
                wait = 5 * (attempt + 1)  # 5s, 10s, 15s
                print(f"      Gemini busy (attempt {attempt + 1}/{max_retries}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                # Non-retryable error — re-raise immediately
                raise
    
    return "ERROR: Gemini is temporarily unavailable. Try again in a few minutes."


# -----------------------------------------------------------------
# DEMO: end-to-end test
# -----------------------------------------------------------------
if __name__ == "__main__":
    test_queries = [
        "I need a lightweight material for marine use",
        "Material for a shaft that must handle 300 MPa stress at 200 degrees C",
        "What material should I use for a food processing tank?",
    ]
    
    for q in test_queries:
        answer = reason_about_query(q, top_k=3)
        print("\n" + "=" * 70)
        print("FINAL ANSWER:")
        print("=" * 70)
        print(answer)
        print()