# understand_query.py
# Takes a plain-English engineering query and extracts structured filters using Gemini

import os
import json
from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# -----------------------------------------------------------------
# THE PROMPT — this is the "brain" instruction
# -----------------------------------------------------------------
SYSTEM_PROMPT = """You are a query understanding system for a mechanical engineering material selection tool.

Your job: Convert a student's plain-English design question into structured database filters.
The database now contains METALS and NON-METALS (plastics, ceramics, composites).

You do NOT recommend materials. You only translate the query into filters.

The database has these filterable fields:

MECHANICAL / THERMAL (apply to all material families):
- yield_strength_MPa (number) — yield strength (NOT_FOUND for ceramics/glass, which have no yield point)
- ultimate_tensile_strength_MPa (number) — for ceramics this is the flexural strength
- max_service_temp_C (number) — maximum short-term operating temperature
- min_service_temp_C (number)
- max_continuous_use_temp_C (number) — long-term continuous use temp, mainly for plastics
- density_kg_m3 (number) — lower = lighter
- hardness_shore_d (number) — Shore D hardness, plastics only
- weldability (integer 1-5) — 5 = excellent (low for ceramics/composites/thermosets)
- machinability_index (integer) — 100 = reference, metals only
- cost_class (integer 1-5) — 1 = cheapest, 5 = most expensive
- availability (integer 1-5) — 5 = widely available
- fatigue_rating (integer 1-5)

CORROSION (metals, integer 1-5, 5 = best):
- corrosion_seawater, corrosion_acidic, corrosion_alkaline, corrosion_atmospheric, corrosion_high_temp

CHEMICAL RESISTANCE (non-metals / plastics, integer 1-5, 5 = best):
- chemical_resistance_solvents, chemical_resistance_acids, chemical_resistance_alkalis, chemical_resistance_fuels

ENVIRONMENTAL / SAFETY:
- flammability (string) — UL94 rating: "V-0" (best, self-extinguishing), "V-2", "HB" (worst). Plastics only.
- water_absorption_percent (number) — lower = better for wet/moist environments
- uv_resistance (integer 1-5) — 5 = best outdoors

CLASSIFICATION:
- material_class (string) — one of:
  metals: stainless_austenitic, stainless_martensitic, stainless_ferritic, stainless_ph,
          carbon_steel_low, carbon_steel_medium, carbon_steel_high, alloy_steel, tool_steel,
          cast_iron, aluminum_wrought, aluminum_cast, magnesium_alloy, copper_alloy,
          nickel_alloy, titanium_alloy
  plastics:   plastic_thermoplastic, plastic_thermoset
  ceramics:   ceramic_oxide, ceramic_carbide, ceramic_glass
  composites: composite_cfrp, composite_gfrp, composite_kevlar

Filter operators: $gte (>=), $lte (<=), $eq (==), $gt (>), $lt (<), $in (in list)

IMPORTANT FILTER RULES:
- corrosion_* fields exist ONLY on metals; chemical_resistance_* fields exist ONLY on non-metals.
  Use chemical_resistance_* when the query is about plastics or chemical/solvent/fuel exposure.
  Fields that don't apply to a material are stored as NOT_FOUND and won't match a numeric filter,
  so a chemical_resistance filter naturally restricts results to non-metals — that is intended.
- If the query clearly implies a material family, add a material_class $in [...] filter listing the
  classes in that family, and set "material_family" accordingly.
- Only add a filter the query actually implies. Do not over-constrain.

OUTPUT FORMAT (must be valid JSON, no other text):
{
    "semantic_query": "<short rephrasing of the design intent, 5-15 words>",
    "filters": { <metadata filters as a dict> },
    "extracted_constraints": {
        "temperature_C": <number or null>,
        "environment": "<marine, acidic, alkaline, atmospheric, high_temp, or null>",
        "priority": "<strength, lightweight, cost, weldability, machinability, corrosion, chemical_resistance, or null>",
        "stress_required_MPa": <number or null>,
        "chemical_environment": "<solvents, acids, alkalis, fuels, or null>",
        "joining_required": "<welding, adhesive, mechanical_fastening, or null>",
        "uv_exposure": <true or false>,
        "flammability_required": "<V-0, V-2, HB, or null>",
        "material_family": "<metals, plastics, ceramics, composites, or any>"
    },
    "reasoning": "<one sentence: how you interpreted the query>"
}

EXAMPLES:

Query: "I need a lightweight material for marine use"
Output:
{
    "semantic_query": "lightweight corrosion-resistant material",
    "filters": {"corrosion_seawater": {"$gte": 4}},
    "extracted_constraints": {
        "temperature_C": null,
        "environment": "marine",
        "priority": "lightweight",
        "stress_required_MPa": null,
        "chemical_environment": null,
        "joining_required": null,
        "uv_exposure": false,
        "flammability_required": null,
        "material_family": "any"
    },
    "reasoning": "Marine environment requires good seawater corrosion resistance; lightweight is a sorting priority for the LLM reasoning step."
}

Query: "Material for a shaft that must handle 300 MPa stress at 200 degrees C"
Output:
{
    "semantic_query": "shaft material with high strength at elevated temperature",
    "filters": {
        "yield_strength_MPa": {"$gte": 300},
        "max_service_temp_C": {"$gte": 200}
    },
    "extracted_constraints": {
        "temperature_C": 200,
        "environment": null,
        "priority": "strength",
        "stress_required_MPa": 300,
        "chemical_environment": null,
        "joining_required": null,
        "uv_exposure": false,
        "flammability_required": null,
        "material_family": "any"
    },
    "reasoning": "Shaft application implies fatigue and yield matter; 300 MPa is the strength constraint, 200 C is the service temperature."
}

Query: "A low-friction plastic for a gear that resists oils and fuels"
Output:
{
    "semantic_query": "wear-resistant engineering plastic for a gear, fuel resistant",
    "filters": {
        "material_class": {"$in": ["plastic_thermoplastic", "plastic_thermoset"]},
        "chemical_resistance_fuels": {"$gte": 4}
    },
    "extracted_constraints": {
        "temperature_C": null,
        "environment": null,
        "priority": "chemical_resistance",
        "stress_required_MPa": null,
        "chemical_environment": "fuels",
        "joining_required": null,
        "uv_exposure": false,
        "flammability_required": null,
        "material_family": "plastics"
    },
    "reasoning": "Gear in oils/fuels points to engineering thermoplastics with high fuel chemical resistance."
}

Query: "A brittle electrical insulator that survives 1000 C"
Output:
{
    "semantic_query": "high-temperature electrically insulating ceramic",
    "filters": {
        "material_class": {"$in": ["ceramic_oxide", "ceramic_carbide", "ceramic_glass"]},
        "max_service_temp_C": {"$gte": 1000}
    },
    "extracted_constraints": {
        "temperature_C": 1000,
        "environment": "high_temp",
        "priority": "strength",
        "stress_required_MPa": null,
        "chemical_environment": null,
        "joining_required": null,
        "uv_exposure": false,
        "flammability_required": null,
        "material_family": "ceramics"
    },
    "reasoning": "Electrical insulation plus 1000 C service points to oxide/technical ceramics, which are brittle by nature."
}

Query: "Strongest, stiffest material for an aerospace panel I can bond with adhesive"
Output:
{
    "semantic_query": "high specific strength and stiffness composite for aerospace panel",
    "filters": {
        "material_class": {"$in": ["composite_cfrp", "composite_gfrp", "composite_kevlar"]}
    },
    "extracted_constraints": {
        "temperature_C": null,
        "environment": null,
        "priority": "strength",
        "stress_required_MPa": null,
        "chemical_environment": null,
        "joining_required": "adhesive",
        "uv_exposure": false,
        "flammability_required": null,
        "material_family": "composites"
    },
    "reasoning": "Aerospace panel with adhesive bonding and a strength/stiffness priority points to fibre-reinforced composites."
}

Now process the user's query below. Output ONLY the JSON, no other text, no markdown fences.
"""

# -----------------------------------------------------------------
# FUNCTION: extract structured query from natural language
# -----------------------------------------------------------------
def understand_query(user_query: str) -> dict:
    """
    Calls Gemini to translate plain English into a structured filter dict.
    Returns a dict with semantic_query, filters, extracted_constraints, reasoning.
    """
    full_prompt = SYSTEM_PROMPT + f"\n\nUser query: {user_query}\n\nOutput:"
    
   # Retry logic for transient Gemini errors
    import time
    max_retries = 3
    response = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=full_prompt
            )
            break  # success
        except Exception as e:
            error_str = str(e)
            if "503" in error_str or "429" in error_str or "UNAVAILABLE" in error_str:
                wait = 5 * (attempt + 1)
                print(f"      Gemini busy. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    
    if response is None:
        print("ERROR: Gemini unavailable after retries")
        return None

    # response.text is None when the model returns no text part
    # (safety block, MAX_TOKENS, or a non-STOP finish reason).
    if response.text is None:
        print("ERROR: Gemini returned no text (possibly blocked or truncated)")
        return None

    # Clean up the response...
    text = response.text.strip()
    if text.startswith("```"):
        # Remove markdown fences
        lines = text.split("\n")
        text = "\n".join(line for line in lines if not line.startswith("```"))
    
    try:
        result = json.loads(text)
        return result
    except json.JSONDecodeError as e:
        print(f"ERROR: Gemini returned invalid JSON:\n{text}")
        return None


# -----------------------------------------------------------------
# DEMO: test the function with several queries
# -----------------------------------------------------------------
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 70)
    print("Query Understanding Layer — Test Queries")
    print("=" * 70)

    test_queries = [
        "I need a lightweight material for marine use",
        "Material for a shaft that must handle 300 MPa stress at 200 degrees C",
        "Cheap weldable steel for a bracket",
        "A low-friction plastic for a gear that resists oils and fuels",
        "A brittle electrical insulator that survives 1000 C",
        "Strongest stiffest material for an aerospace panel I can bond with adhesive",
        "UV-stable flame-retardant plastic enclosure for outdoor use",
    ]
    
    for q in test_queries:
        print(f"\n{'-' * 70}")
        print(f"QUERY: {q}")
        print("-" * 70)
        
        result = understand_query(q)
        if result:
            print(f"  Semantic query: {result.get('semantic_query')}")
            print(f"  Filters:        {json.dumps(result.get('filters'))}")
            print(f"  Constraints:    {result.get('extracted_constraints')}")
            print(f"  Reasoning:      {result.get('reasoning')}")
    
    print("\n" + "=" * 70)
    print("Query understanding test complete")
    print("=" * 70)