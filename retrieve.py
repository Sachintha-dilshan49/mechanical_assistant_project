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
# FUNCTION 1: Search materials by semantic query + metadata filter
# =================================================================
def find_materials(query_text, filters=None, top_k=3):
    """
    Search ChromaDB for materials matching a text query and optional filters.
    Filters with multiple conditions are auto-wrapped with $and for ChromaDB.
    """
    print(f"\n[Search] '{query_text}'")
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
        meta = results['metadatas'][0][i]
        materials.append({
            "material_id":              mat_id,
            "common_name":              meta.get('common_name', ''),
            "material_class":           meta.get('material_class', ''),
            "condition":                meta.get('condition', ''),
            "yield_strength_MPa":       meta.get('yield_strength_MPa'),
            "ultimate_tensile_strength_MPa": meta.get('ultimate_tensile_strength_MPa'),
            "max_service_temp_C":       meta.get('max_service_temp_C'),
            "min_service_temp_C":       meta.get('min_service_temp_C'),
            "corrosion_seawater":       meta.get('corrosion_seawater'),
            "corrosion_acidic":         meta.get('corrosion_acidic'),
            "corrosion_alkaline":       meta.get('corrosion_alkaline'),
            "corrosion_atmospheric":    meta.get('corrosion_atmospheric'),
            "corrosion_high_temp":      meta.get('corrosion_high_temp'),
            "weldability":              meta.get('weldability'),
            "machinability_index":      meta.get('machinability_index'),
            "cost_class":               meta.get('cost_class'),
            "availability":             meta.get('availability'),
            "fatigue_rating":           meta.get('fatigue_rating'),
            "stress_table_id":          meta.get('stress_table_id', ''),
            "sources":                  meta.get('sources', ''),
            "description_text":         results['documents'][0][i],
            "relevance_score":          1 - results['distances'][0][i],
            "density_kg_m3":            meta.get('density_kg_m3'),
            "elastic_modulus_GPa":      meta.get('elastic_modulus_GPa'),
            "key_warnings":             meta.get('key_warnings', ''),
            "typical_applications":     meta.get('typical_applications', ''),
            "weldability_notes":        meta.get('weldability_notes', ''),
            "approx_cost_usd_per_kg":   meta.get('approx_cost_usd_per_kg', ''),
        })
    
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
    print("=" * 60)
    print("Retrieval Layer — Test Queries")
    print("=" * 60)
    
    # --------------------------------------------------
    # Test 1: Pure semantic search
    # --------------------------------------------------
    print("\n--- Test 1: Plain English search ---")
    results = find_materials("corrosion-resistant material for marine use")
    for mat in results:
        print(f"  → {mat['common_name']} (relevance: {mat['relevance_score']:.2f})")
        print(f"     Sources: {mat['sources']}")
    
    # --------------------------------------------------
    # Test 2: Semantic search + metadata filter
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