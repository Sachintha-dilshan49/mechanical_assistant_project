"""
load_v1_data.py
Loads the v1 dataset (10 materials + stress data) from CSV files directly into 
ChromaDB and SQLite. Use this instead of typing into the Excel spreadsheet.

USAGE:
    1. Place materials.csv and stress_temperature.csv in your data/ folder
    2. Activate venv: venv\\Scripts\\activate
    3. Run: python load_v1_data.py
"""

import pandas as pd
import sqlite3
import chromadb
from pathlib import Path

# File paths
MATERIALS_CSV = "data/materials.csv"
STRESS_CSV = "data/stress_temperature.csv"
DB_FILE = "data/materials.db"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "materials"

# Sentinel used in the CSV for properties that don't apply to a given material
# (e.g. Shore D hardness on a metal, machinability on a ceramic, yield on glass).
NOT_FOUND = "NOT_FOUND"

# Every material_class present in the v3 dataset (metals + non-metals).
# The loader warns if a row uses a class outside this set, but still loads it.
KNOWN_CLASSES = {
    "stainless_austenitic", "stainless_martensitic", "stainless_ferritic",
    "stainless_ph", "carbon_steel_low", "carbon_steel_medium",
    "carbon_steel_high", "alloy_steel", "tool_steel", "cast_iron",
    "aluminum_wrought", "aluminum_cast", "magnesium_alloy", "copper_alloy",
    "nickel_alloy", "titanium_alloy",
    "plastic_thermoplastic", "plastic_thermoset",
    "ceramic_oxide", "ceramic_carbide", "ceramic_glass",
    "composite_cfrp", "composite_gfrp", "composite_kevlar",
}


def meta_num(val):
    """Float for a numeric value, or the string 'NOT_FOUND' when the property
    does not apply. Storing the sentinel as a string (not 0) keeps ChromaDB
    $gte/$lte range filters from matching rows where the value is missing."""
    s = str(val).strip()
    if s == "" or s.upper() == "NOT_FOUND":
        return NOT_FOUND
    try:
        return float(s)
    except ValueError:
        return NOT_FOUND


def meta_str(val):
    """Clean string. Empty cells stay ''; an explicit 'NOT_FOUND' is preserved
    so the UI can render it as N/A (e.g. flammability on a metal)."""
    return str(val).strip()


print("=" * 60)
print("Loading v1 dataset into ChromaDB + SQLite")
print("=" * 60)

Path("data").mkdir(exist_ok=True)

# ============================================================
# PART A — Load stress data into SQLite
# ============================================================
print("\n[A] Loading stress-temperature data into SQLite...")

stress_df = pd.read_csv(STRESS_CSV)
print(f"    Read {len(stress_df)} stress rows from CSV")

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS allowable_stress")
cursor.execute("""
    CREATE TABLE allowable_stress (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        material_id     TEXT NOT NULL,
        stress_table_id TEXT NOT NULL,
        temperature_C   INTEGER NOT NULL,
        stress_MPa      REAL NOT NULL,
        source          TEXT NOT NULL,
        notes           TEXT
    )
""")
cursor.execute("""
    CREATE INDEX idx_stress_lookup 
    ON allowable_stress(stress_table_id, temperature_C)
""")

for _, row in stress_df.iterrows():
    cursor.execute("""
        INSERT INTO allowable_stress 
        (material_id, stress_table_id, temperature_C, stress_MPa, source, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        row['material_id'],
        row['stress_table_id'],
        int(row['temperature_C']),
        float(row['stress_MPa']),
        row['source'],
        row.get('notes', '') if pd.notna(row.get('notes', '')) else ''
    ))

conn.commit()
conn.close()
print(f"    Inserted {len(stress_df)} rows into SQLite")

# ============================================================
# PART B — Load materials into ChromaDB
# ============================================================
print("\n[B] Loading materials into ChromaDB...")

# Optional bulk dataset (Kaggle mechanical-properties dump), already cleaned and
# mapped into our schema by build_kaggle_dataset.py. Loaded into the same pool
# when present; skipped silently when absent.
KAGGLE_CSV = "data/kaggle_clean.csv"


def build_metadata(row):
    """Map one CSV row (all string cells) to a ChromaDB metadata dict."""
    return {
        # --- identity / classification ---
        "common_name":                   meta_str(row['common_name']),
        "material_class":                meta_str(row['material_class']),
        "condition":                     meta_str(row['condition']),
        "uns_number":                    meta_str(row['uns_number']),
        "aisi_grade":                    meta_str(row['aisi_grade']),
        # --- mechanical ---
        "yield_strength_MPa":            meta_num(row['yield_strength_MPa']),
        "ultimate_tensile_strength_MPa": meta_num(row['ultimate_tensile_strength_MPa']),
        "elongation_percent":            meta_num(row['elongation_percent']),
        "hardness_HB":                   meta_num(row['hardness_HB']),
        "hardness_shore_d":              meta_num(row['hardness_shore_d']),
        "fatigue_limit_MPa":             meta_num(row['fatigue_limit_MPa']),
        "elastic_modulus_GPa":           meta_num(row['elastic_modulus_GPa']),
        "density_kg_m3":                 meta_num(row['density_kg_m3']),
        # --- thermal ---
        "max_service_temp_C":            meta_num(row['max_service_temp_C']),
        "min_service_temp_C":            meta_num(row['min_service_temp_C']),
        "max_continuous_use_temp_C":     meta_num(row['max_continuous_use_temp_C']),
        # --- corrosion (1-5, metals) ---
        "corrosion_seawater":            meta_num(row['corrosion_seawater']),
        "corrosion_acidic":              meta_num(row['corrosion_acidic']),
        "corrosion_alkaline":            meta_num(row['corrosion_alkaline']),
        "corrosion_atmospheric":         meta_num(row['corrosion_atmospheric']),
        "corrosion_high_temp":           meta_num(row['corrosion_high_temp']),
        # --- chemical resistance (1-5, non-metals) ---
        "chemical_resistance_solvents":  meta_num(row['chemical_resistance_solvents']),
        "chemical_resistance_acids":     meta_num(row['chemical_resistance_acids']),
        "chemical_resistance_alkalis":   meta_num(row['chemical_resistance_alkalis']),
        "chemical_resistance_fuels":     meta_num(row['chemical_resistance_fuels']),
        # --- joining / fabrication ---
        "weldability":                   meta_num(row['weldability']),
        "weldability_notes":             meta_str(row['weldability_notes']),
        "joining_method":                meta_str(row['joining_method']),
        "machinability_index":           meta_num(row['machinability_index']),
        # --- environmental / safety ---
        "flammability":                  meta_str(row['flammability']),
        "water_absorption_percent":      meta_num(row['water_absorption_percent']),
        "uv_resistance":                 meta_num(row['uv_resistance']),
        # --- commercial ---
        "cost_class":                    meta_num(row['cost_class']),
        "approx_cost_usd_per_kg":        meta_str(row['approx_cost_usd_per_kg']),
        "availability":                  meta_num(row['availability']),
        "stock_forms":                   meta_str(row['stock_forms']),
        "fatigue_rating":                meta_num(row['fatigue_rating']),
        # --- text / provenance ---
        "typical_applications":          meta_str(row['typical_applications']),
        "key_warnings":                  meta_str(row['key_warnings']),
        "sources":                       meta_str(row['sources']),
        "stress_table_id":               meta_str(row['stress_table_id']),
    }


def read_rows(path):
    """Read a materials CSV as all-strings, dropping rows with a blank id."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df[df['material_id'].str.strip() != '']


client = chromadb.PersistentClient(path=CHROMA_DIR)

try:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"    Deleted existing '{COLLECTION_NAME}' collection")
except Exception:
    print(f"    No existing collection (first run)")

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"description": "Engineering materials for design selection"}
)

# Curated 66 always load; the Kaggle bulk dataset loads too when present.
sources_to_load = [("curated", MATERIALS_CSV)]
if Path(KAGGLE_CSV).exists():
    sources_to_load.append(("kaggle bulk", KAGGLE_CSV))

ids = []
documents = []
metadatas = []
seen_ids = set()

for label, path in sources_to_load:
    df = read_rows(path)
    added = 0
    for _, row in df.iterrows():
        mid = str(row['material_id']).strip()
        if mid in seen_ids:          # never add the same id twice
            continue
        seen_ids.add(mid)
        ids.append(mid)
        documents.append(row['description_text'])
        metadatas.append(build_metadata(row))
        added += 1
    print(f"    Read {added} rows from {path} ({label})")

# Allowlist sanity check — load everything, but flag any unexpected class so a
# typo in the CSV (e.g. "plastic_thermoplastik") surfaces instead of silently
# breaking family filters downstream.
seen_classes = {m['material_class'] for m in metadatas}
unknown = seen_classes - KNOWN_CLASSES
if unknown:
    print(f"    WARNING: unexpected material_class values not in allowlist: {sorted(unknown)}")
print(f"    Material classes loaded ({len(seen_classes)}): {', '.join(sorted(seen_classes))}")

print(f"    Embedding {len(ids)} documents and storing... (first run downloads model)")
# Batch the add — the Kaggle merge pushes this to ~2,000 rows, and batching keeps
# us under ChromaDB's max batch size and shows progress on slow CPU embedding.
BATCH = 500
for start in range(0, len(ids), BATCH):
    end = start + BATCH
    collection.add(
        ids=ids[start:end],
        documents=documents[start:end],
        metadatas=metadatas[start:end],
    )
    print(f"      stored {min(end, len(ids))}/{len(ids)}")
print(f"    Stored {collection.count()} vectors in ChromaDB")

# ============================================================
# VERIFICATION
# ============================================================
print("\n" + "=" * 60)
print("Quick verification queries")
print("=" * 60)

print("\n--- Test: 'lightweight marine material' ---")
results = collection.query(query_texts=["lightweight marine material"], n_results=3)
for i, mat_id in enumerate(results['ids'][0]):
    name = results['metadatas'][0][i]['common_name']
    distance = results['distances'][0][i]
    print(f"  {i+1}. {name}  (distance: {distance:.3f})")

print("\n--- Test: 'shaft material for gearbox' ---")
results = collection.query(query_texts=["shaft material for gearbox"], n_results=3)
for i, mat_id in enumerate(results['ids'][0]):
    name = results['metadatas'][0][i]['common_name']
    distance = results['distances'][0][i]
    print(f"  {i+1}. {name}  (distance: {distance:.3f})")

print("\n--- Test: 'good corrosion resistance' WHERE corrosion_seawater >= 4 ---")
results = collection.query(
    query_texts=["good corrosion resistance"],
    n_results=5,
    where={"corrosion_seawater": {"$gte": 4}}
)
for i, mat_id in enumerate(results['ids'][0]):
    name = results['metadatas'][0][i]['common_name']
    sea = results['metadatas'][0][i]['corrosion_seawater']
    print(f"  {i+1}. {name}  (corrosion_seawater: {sea}/5)")

print("\n--- Test: 'lightweight plastic for a gear' WHERE material_class IN plastics ---")
results = collection.query(
    query_texts=["lightweight low-friction plastic for a gear"],
    n_results=3,
    where={"material_class": {"$in": ["plastic_thermoplastic", "plastic_thermoset"]}}
)
for i, mat_id in enumerate(results['ids'][0]):
    meta = results['metadatas'][0][i]
    print(f"  {i+1}. {meta['common_name']}  (Shore D: {meta['hardness_shore_d']}, class: {meta['material_class']})")

print("\n" + "=" * 60)
print("v1 database loaded successfully")
print(f"  ChromaDB: {CHROMA_DIR}")
print(f"  SQLite:   {DB_FILE}")
print("=" * 60)