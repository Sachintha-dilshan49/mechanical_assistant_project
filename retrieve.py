# retrieve.py
# The retrieval layer — combines ChromaDB (semantic) and SQLite (exact) queries

import chromadb
import sqlite3
from pathlib import Path

# File paths
CHROMA_DIR = "data/chroma_db"
DB_FILE = "data/materials.db"
COLLECTION_NAME = "materials"

# -----------------------------------------------------------------
# Connect to both databases (once, at module load)
# -----------------------------------------------------------------
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)


# =================================================================
# Material-family keyword mapping
# Maps loose words in a query ("plastic", "ceramic", "carbon fibre") to the
# concrete material_class values in the database. Used as a safety net so that
# even when the LLM doesn't emit a material_class filter (or is unavailable),
# a query for "plastic gear" is still constrained to polymer classes.
# =================================================================
def infer_family_classes(query_text):
    """Return a list of material_class values implied by family keywords in the
    query, or None if the query doesn't name a material family."""
    t = (query_text or "").lower()
    classes = []

    def add(*cls):
        for c in cls:
            if c not in classes:
                classes.append(c)

    # --- plastics / polymers ---
    if any(k in t for k in ("plastic", "polymer", "thermoplastic")):
        add("plastic_thermoplastic", "plastic_thermoset")
    if "thermoset" in t:
        add("plastic_thermoset")

    # --- ceramics ---
    if "ceramic" in t:
        add("ceramic_oxide", "ceramic_carbide", "ceramic_glass")

    # --- composites (generic word covers all three families) ---
    if "composite" in t:
        add("composite_cfrp", "composite_gfrp", "composite_kevlar")
    if any(k in t for k in ("carbon fibre", "carbon fiber", "cfrp")):
        add("composite_cfrp")
    if any(k in t for k in ("fiberglass", "fibreglass", "gfrp", "glass fibre", "glass fiber")):
        add("composite_gfrp")
    if any(k in t for k in ("kevlar", "aramid")):
        add("composite_kevlar")

    return classes or None


# =================================================================
# FUNCTION 1: Search materials by semantic query + metadata filter
# =================================================================
def find_materials(query_text, filters=None, top_k=3):
    """
    Search ChromaDB for materials matching a text query and optional filters.
    Filters with multiple conditions are auto-wrapped with $and for ChromaDB.
    """
    print(f"\n[Search] '{query_text}'")

    # Work on a copy so we never mutate the caller's filter dict.
    filters = dict(filters) if filters else {}

    # If the caller hasn't pinned a material_class but the query names a family
    # (plastic / ceramic / composite / carbon fibre ...), constrain to it.
    if "material_class" not in filters:
        fam = infer_family_classes(query_text)
        if fam:
            filters["material_class"] = {"$in": fam}
            print(f"[Family] query implies material_class in {fam}")

    if filters:
        print(f"[Filter] {filters}")

    # Normalize filters: ChromaDB requires $and/$or for multi-condition filters
    chroma_filter = None
    if filters:
        if len(filters) == 1:
            # Single filter — use as-is
            chroma_filter = filters
        else:
            # Multiple filters — wrap in $and
            chroma_filter = {"$and": [{k: v} for k, v in filters.items()]}
    
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        where=chroma_filter
    )
    
    # Format the results into a clean list of dicts
    materials = []
    if not results['ids'][0]:
        print("[Result] No materials matched the filters")
        return materials
    
    for i, mat_id in enumerate(results['ids'][0]):
        # Carry every stored metadata field through verbatim (including all the
        # v3 non-metal fields: hardness_shore_d, chemical_resistance_*,
        # joining_method, flammability, water_absorption_percent, uv_resistance,
        # max_continuous_use_temp_C, ...). Numeric fields are floats; fields
        # that don't apply to this material are the string "NOT_FOUND".
        mat = dict(results['metadatas'][0][i])
        mat["material_id"] = mat_id
        mat["description_text"] = results['documents'][0][i]
        mat["relevance_score"] = 1 - results['distances'][0][i]
        materials.append(mat)

    return materials


# =================================================================
# FUNCTION 2: Look up exact allowable stress at a given temperature
# =================================================================
def get_allowable_stress(stress_table_id, temp_C):
    """
    Returns allowable stress (MPa) for a material at the given temperature.
    Interpolates linearly between data points if needed.
    
    stress_table_id: e.g., "316L_annealed"
    temp_C:          temperature in degrees Celsius
    
    Returns dict: {stress_MPa, temperature_C, interpolated, source}
    Returns None if material not in database or temp out of range.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Fetch all data points for this material, sorted by temperature
    cursor.execute("""
        SELECT temperature_C, stress_MPa, source
        FROM allowable_stress
        WHERE stress_table_id = ?
        ORDER BY temperature_C
    """, (stress_table_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return None
    
    # Check temperature range
    min_temp = rows[0][0]
    max_temp = rows[-1][0]
    
    if temp_C < min_temp or temp_C > max_temp:
        return {
            "error": f"Temperature {temp_C}°C is outside the available range ({min_temp}–{max_temp}°C)",
            "stress_MPa": None
        }
    
    # Look for exact match first
    for t, s, src in rows:
        if t == temp_C:
            return {
                "stress_MPa": s,
                "temperature_C": temp_C,
                "interpolated": False,
                "source": src
            }
    
    # No exact match — interpolate linearly between the two nearest points
    for i in range(len(rows) - 1):
        t1, s1, src = rows[i]
        t2, s2, _ = rows[i + 1]
        if t1 < temp_C < t2:
            # Linear interpolation: y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
            interpolated_stress = s1 + (s2 - s1) * (temp_C - t1) / (t2 - t1)
            return {
                "stress_MPa": round(interpolated_stress, 1),
                "temperature_C": temp_C,
                "interpolated": True,
                "source": src,
                "between": f"{t1}°C ({s1} MPa) and {t2}°C ({s2} MPa)"
            }
    
    return None


# =================================================================
# DEMO: Run some example queries
# =================================================================
if __name__ == "__main__":
    # The demo prints unicode (arrows, °C). Force UTF-8 so it doesn't crash on
    # a legacy cp1252 Windows console. Only affects direct CLI runs, not the app.
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 60)
    print("Retrieval Layer — Test Queries")
    print("=" * 60)
    
    # --------------------------------------------------
    # Test 1: v3 validation queries (metals + non-metals)
    # --------------------------------------------------
    validation_queries = [
        "lightweight plastic for a gear",
        "corrosion resistant material for chemical storage",
        "high strength structural material for aerospace",
        "brittle material for electrical insulation at 1000 C",
    ]
    print("\n--- Test 1: v3 semantic search (family inference) ---")
    for q in validation_queries:
        results = find_materials(q, top_k=3)
        names = ", ".join(f"{m['common_name']}" for m in results)
        print(f"  Q: {q}\n     → {names}")

    # --------------------------------------------------
    # Test 2: Semantic search + explicit metadata filter
    # --------------------------------------------------
    print("\n--- Test 2: Search with filter (corrosion_seawater >= 4) ---")
    results = find_materials(
        "material for chemical processing equipment",
        filters={"corrosion_seawater": {"$gte": 4}}
    )
    for mat in results:
        print(f"  → {mat['common_name']} (corrosion: {mat['corrosion_seawater']}/5)")

    # --------------------------------------------------
    # Test 3: Exact allowable stress lookup (in-table)
    # --------------------------------------------------
    print("\n--- Test 3: Stress lookup at 200°C (exact match) ---")
    stress = get_allowable_stress("316L_annealed", 200)
    if stress:
        print(f"  Allowable stress at 200°C: {stress['stress_MPa']} MPa")
        print(f"  Interpolated: {stress['interpolated']}")
        print(f"  Source: {stress['source']}")
    
    # --------------------------------------------------
    # Test 4: Stress lookup requiring interpolation
    # --------------------------------------------------
    print("\n--- Test 4: Stress lookup at 275°C (interpolated) ---")
    stress = get_allowable_stress("316L_annealed", 275)
    if stress:
        print(f"  Allowable stress at 275°C: {stress['stress_MPa']} MPa")
        print(f"  Interpolated: {stress['interpolated']}")
        if stress.get('between'):
            print(f"  Between: {stress['between']}")
    
    # --------------------------------------------------
    # Test 5: Stress lookup out of range (error case)
    # --------------------------------------------------
    print("\n--- Test 5: Stress lookup at 600°C (out of range) ---")
    stress = get_allowable_stress("316L_annealed", 600)
    if stress and stress.get('error'):
        print(f"  Error: {stress['error']}")
    
    print("\n" + "=" * 60)
    print("Retrieval tests complete")
    print("=" * 60)