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

You do NOT recommend materials. You only translate the query into filters.

The database has these filterable fields:
- yield_strength_MPa (integer) — material yield strength
- ultimate_tensile_strength_MPa (integer)
- max_service_temp_C (integer) — maximum operating temperature
- min_service_temp_C (integer)
- corrosion_seawater (integer 1-5) — 5 = best for marine
- corrosion_acidic (integer 1-5)
- corrosion_alkaline (integer 1-5)
- corrosion_atmospheric (integer 1-5)
- corrosion_high_temp (integer 1-5)
- weldability (integer 1-5) — 5 = excellent
- machinability_index (integer) — 100 = AISI 1212 reference
- cost_class (integer 1-5) — 1 = cheapest, 5 = most expensive
- availability (integer 1-5) — 5 = widely available
- fatigue_rating (integer 1-5)
- material_class (string) — one of: stainless_austenitic, stainless_martensitic, stainless_ferritic, stainless_ph, carbon_steel_low, carbon_steel_medium, carbon_steel_high, alloy_steel, tool_steel, aluminum_wrought, aluminum_cast, copper_alloy, cast_iron, titanium_alloy

Filter operators: $gte (>=), $lte (<=), $eq (==), $gt (>), $lt (<), $in (in list)

OUTPUT FORMAT (must be valid JSON, no other text):
{
    "semantic_query": "<short rephrasing of the design intent, 5-15 words>",
    "filters": { <metadata filters as a dict> },
    "extracted_constraints": {
        "temperature_C": <number or null>,
        "environment": "<marine, acidic, alkaline, atmospheric, high_temp, or null>",
        "priority": "<strength, lightweight, cost, weldability, machinability, corrosion, or null>",
        "stress_required_MPa": <number or null>
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
        "stress_required_MPa": null
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
        "stress_required_MPa": 300
    },
    "reasoning": "Shaft application implies fatigue and yield matter; 300 MPa is the strength constraint, 200 C is the service temperature."
}

Query: "Cheap weldable steel for a bracket"
Output:
{
    "semantic_query": "low-cost weldable steel for structural bracket",
    "filters": {
        "weldability": {"$gte": 4},
        "cost_class": {"$lte": 2},
        "material_class": {"$in": ["carbon_steel_low", "carbon_steel_medium"]}
    },
    "extracted_constraints": {
        "temperature_C": null,
        "environment": null,
        "priority": "cost",
        "stress_required_MPa": null
    },
    "reasoning": "Bracket implies low stress so material class is carbon steel; cheap and weldable are explicit constraints."
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
    print("=" * 70)
    print("Query Understanding Layer — Test Queries")
    print("=" * 70)
    
    test_queries = [
        "I need a lightweight material for marine use",
        "Material for a shaft that must handle 300 MPa stress at 200 degrees C",
        "Cheap weldable steel for a bracket",
        "Corrosion-resistant material for food processing equipment",
        "Aerospace material for wing skin",
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