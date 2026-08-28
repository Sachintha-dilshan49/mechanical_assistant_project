"""
reindex.py
Rebuilds the ChromaDB search index from the SQLite record store.

ChromaDB is derived data: it holds embeddings of each material's description so
semantic search works. Deleting data/chroma_db loses nothing - run this and it
comes back. Run it any time materials are added or edited in SQLite.

USAGE:
    py reindex.py
"""

import shutil
from pathlib import Path

import chromadb

import db

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "materials"
BATCH = 500

print("=" * 60)
print("Rebuilding ChromaDB search index from SQLite")
print("=" * 60)

# -----------------------------------------------------------------
# STEP 1: Read the source of truth
# -----------------------------------------------------------------
conn = db.connect()
materials = db.fetch_materials(conn)
conn.close()

if not materials:
    print(f"\n    No materials in {db.DB_FILE}. Run: py init_db.py")
    raise SystemExit(1)

print(f"\n[1/3] Read {len(materials)} materials from {db.DB_FILE}")

# -----------------------------------------------------------------
# STEP 2: Start from an empty index.
# Chroma leaves orphaned segment folders behind when a collection is deleted, so
# removing the directory outright keeps the index from growing with dead vectors.
# -----------------------------------------------------------------
print(f"\n[2/3] Clearing {CHROMA_DIR}")
if Path(CHROMA_DIR).exists():
    shutil.rmtree(CHROMA_DIR)

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"description": "Engineering materials for design selection"},
)

# -----------------------------------------------------------------
# STEP 3: Embed and store.
# NULL becomes the "NOT_FOUND" sentinel here (Chroma metadata can't hold nulls),
# which is exactly the value the retrieval and UI layers already read as N/A.
# -----------------------------------------------------------------
ids = [m["material_id"] for m in materials]
documents = [m["description_text"] for m in materials]
metadatas = [db.to_chroma_metadata(m) for m in materials]

print(f"\n[3/3] Embedding {len(ids)} documents... (first run downloads the model)")
for start in range(0, len(ids), BATCH):
    end = start + BATCH
    collection.add(
        ids=ids[start:end],
        documents=documents[start:end],
        metadatas=metadatas[start:end],
    )
    print(f"      stored {min(end, len(ids))}/{len(ids)}")

count = collection.count()
print(f"\n      Index holds {count} vectors")

# -----------------------------------------------------------------
# VERIFICATION — a semantic query, a numeric filter and a class filter, so a
# broken metadata conversion shows up here instead of in the app.
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("Verification")
print("=" * 60)

res = collection.query(query_texts=["good corrosion resistance"], n_results=3,
                       where={"corrosion_seawater": {"$gte": 4}})
print("\n  numeric filter (corrosion_seawater >= 4):")
for i, _ in enumerate(res["ids"][0]):
    m = res["metadatas"][0][i]
    print(f"    - {m['common_name']}  ({m['corrosion_seawater']}/5)")

res = collection.query(query_texts=["lightweight low-friction plastic for a gear"],
                       n_results=3,
                       where={"material_class": {"$in": ["plastic_thermoplastic",
                                                         "plastic_thermoset"]}})
print("\n  class filter (plastics):")
for i, _ in enumerate(res["ids"][0]):
    m = res["metadatas"][0][i]
    print(f"    - {m['common_name']}  (Shore D: {m['hardness_shore_d']})")

print(f"\n  NOT_FOUND sentinel preserved: "
      f"{res['metadatas'][0][0]['hardness_HB'] == db.NOT_FOUND}")

# data_confidence is what the card uses to say whether a number came from an ASME
# table or is my estimate. It rides along with every other column, so the only way
# it can go missing is a row written before the column existed - check it here
# rather than shipping an index where the claim has silently disappeared.
conf = [m.get("data_confidence") for m in metadatas]
missing = [i for i, c in zip(ids, conf) if c in (None, db.NOT_FOUND)]
print(f"  data_confidence indexed: {len(conf) - len(missing)}/{len(conf)}")
for value in db.DATA_CONFIDENCE_VALUES:
    print(f"    {value:<24} {conf.count(value)}")
if missing:
    print(f"    WARNING: no confidence on {len(missing)} row(s): {missing[:5]}")
    print("    Run: py migrate_add_confidence.py")

print("\n" + "=" * 60)
print(f"Index rebuilt: {count} vectors in {CHROMA_DIR}")
print("=" * 60)
