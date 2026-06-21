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

materials_df = pd.read_csv(MATERIALS_CSV)
materials_df = materials_df.dropna(subset=['material_id'])
print(f"    Read {len(materials_df)} materials from CSV")

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

ids = []
documents = []
metadatas = []

for _, row in materials_df.iterrows():
    ids.append(row['material_id'])
    documents.append(row['description_text'])
    
    metadata = {
        "common_name":                  str(row['common_name']),
        "material_class":               str(row['material_class']),
        "condition":                    str(row['condition']),
        "yield_strength_MPa":           int(row['yield_strength_MPa']),
        "ultimate_tensile_strength_MPa": int(row['ultimate_tensile_strength_MPa']),
        "max_service_temp_C":           int(row['max_service_temp_C']),
        "min_service_temp_C":           int(row['min_service_temp_C']),
        "corrosion_seawater":           int(row['corrosion_seawater']),
        "corrosion_acidic":             int(row['corrosion_acidic']),
        "corrosion_alkaline":           int(row['corrosion_alkaline']),
        "corrosion_atmospheric":        int(row['corrosion_atmospheric']),
        "corrosion_high_temp":          int(row['corrosion_high_temp']),
        "weldability":                  int(row['weldability']),
        "machinability_index":          int(row['machinability_index']),
        "cost_class":                   int(row['cost_class']),
        "availability":                 int(row['availability']),
        "fatigue_rating":               int(row['fatigue_rating']),
        "sources":                      str(row['sources']),
        "stress_table_id":              str(row['stress_table_id']) if pd.notna(row['stress_table_id']) else '',
    }
    metadatas.append(metadata)

print("    Embedding text and storing... (first run downloads model)")
collection.add(ids=ids, documents=documents, metadatas=metadatas)
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

print("\n" + "=" * 60)
print("v1 database loaded successfully")
print(f"  ChromaDB: {CHROMA_DIR}")
print(f"  SQLite:   {DB_FILE}")
print("=" * 60)