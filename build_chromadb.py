# build_chromadb.py
# Reads the Materials sheet and loads each material into ChromaDB as an embedded vector

import pandas as pd
import chromadb
from pathlib import Path

# File paths
EXCEL_FILE = "data/materials_database_template.xlsx"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "materials"

print("=" * 60)
print("Building ChromaDB from spreadsheet")
print("=" * 60)

# -----------------------------------------------------------------
# STEP 1: Read the spreadsheet
# -----------------------------------------------------------------
print("\n[1/5] Reading Materials sheet from Excel...")

materials_df = pd.read_excel(EXCEL_FILE, sheet_name="Materials")
materials_df = materials_df.dropna(subset=['material_id'])

print(f"      Loaded {len(materials_df)} materials")

# -----------------------------------------------------------------
# STEP 2: Connect to ChromaDB (creates the folder if missing)
# -----------------------------------------------------------------
print(f"\n[2/5] Setting up ChromaDB at: {CHROMA_DIR}")

Path("data").mkdir(exist_ok=True)

# Persistent client = saves to disk (not just memory)
client = chromadb.PersistentClient(path=CHROMA_DIR)

# Delete the collection if it already exists (so we start fresh)
try:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"      Deleted existing '{COLLECTION_NAME}' collection")
except Exception:
    print(f"      No existing collection to delete (first run)")

# Create a fresh collection
collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"description": "Engineering materials for design selection"}
)

print(f"      Created collection: {COLLECTION_NAME}")

# -----------------------------------------------------------------
# STEP 3: Prepare data for ChromaDB
# -----------------------------------------------------------------
print("\n[3/5] Preparing data...")

ids = []           # list of material_ids
documents = []     # list of description_texts (these get embedded)
metadatas = []     # list of metadata dicts (filterable fields)

for index, row in materials_df.iterrows():
    ids.append(row['material_id'])
    documents.append(row['description_text'])
    
    # Build metadata dict (ChromaDB only accepts strings, ints, floats, bools)
    metadata = {
        "common_name":                str(row['common_name']),
        "material_class":             str(row['material_class']),
        "condition":                  str(row['condition']),
        "yield_strength_MPa":         int(row['yield_strength_MPa']),
        "ultimate_tensile_strength_MPa": int(row['ultimate_tensile_strength_MPa']),
        "max_service_temp_C":         int(row['max_service_temp_C']),
        "min_service_temp_C":         int(row['min_service_temp_C']),
        "corrosion_seawater":         int(row['corrosion_seawater']),
        "corrosion_acidic":           int(row['corrosion_acidic']),
        "corrosion_alkaline":         int(row['corrosion_alkaline']),
        "corrosion_atmospheric":      int(row['corrosion_atmospheric']),
        "corrosion_high_temp":        int(row['corrosion_high_temp']),
        "weldability":                int(row['weldability']),
        "machinability_index":        int(row['machinability_index']),
        "cost_class":                 int(row['cost_class']),
        "availability":               int(row['availability']),
        "fatigue_rating":             int(row['fatigue_rating']),
        "sources":                    str(row['sources']),
        "stress_table_id":            str(row['stress_table_id']) if pd.notna(row.get('stress_table_id', '')) else '',
    }
    metadatas.append(metadata)

print(f"      Prepared {len(ids)} entries")

# -----------------------------------------------------------------
# STEP 4: Add to ChromaDB (auto-embeds the documents)
# -----------------------------------------------------------------
print("\n[4/5] Embedding text and storing in ChromaDB...")
print("      (First run downloads the embedding model — may take 1-2 min)")

collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

print(f"      Stored {collection.count()} vectors in ChromaDB")

# -----------------------------------------------------------------
# STEP 5: Verification — search for a test query
# -----------------------------------------------------------------
print("\n[5/5] Test search: 'corrosion-resistant material for marine use'")

results = collection.query(
    query_texts=["corrosion-resistant material for marine use"],
    n_results=3
)

print("\n      Top matches:")
print("      " + "-" * 50)

for i, mat_id in enumerate(results['ids'][0]):
    name = results['metadatas'][0][i]['common_name']
    distance = results['distances'][0][i]
    print(f"      {i+1}. {name}  (id={mat_id}, distance={distance:.3f})")

print("\n" + "=" * 60)
print("ChromaDB built successfully")
print(f"Saved to: {CHROMA_DIR}")
print("=" * 60)