"""
init_db.py
Seeds the SQLite record store from the curated CSV files. Run this once.

SQLite is the source of truth for materials, so this script will NOT overwrite a
table that already has rows — otherwise a habitual re-run would silently delete
materials you added later from a PDF or by hand. Pass --force to reseed anyway.

USAGE:
    py init_db.py            # first-time setup, or top up an empty table
    py init_db.py --force    # wipe and reseed from the CSVs (destructive)

After this, run:  py reindex.py
"""

import sys
import pandas as pd
from pathlib import Path

import db

MATERIALS_CSV = "data/materials.csv"
STRESS_CSV = "data/stress_temperature.csv"

force = "--force" in sys.argv

print("=" * 60)
print("Seeding SQLite record store")
print("=" * 60)

Path("data").mkdir(exist_ok=True)

conn = db.connect()
db.create_schema(conn)
print(f"    Schema ready in {db.DB_FILE}")

# -----------------------------------------------------------------
# Guard: never silently destroy data that is already in the store.
# -----------------------------------------------------------------
existing = db.count_materials(conn)
if existing and not force:
    print(f"\n    materials table already holds {existing} rows - nothing to do.")
    print("    SQLite is the source of truth now, so seeding again would discard")
    print("    anything added since. Re-run with --force if that is what you want.")
    print("\n    To rebuild the search index from these rows: py reindex.py")
    conn.close()
    sys.exit(0)

if existing and force:
    print(f"\n    --force: deleting {existing} existing materials")
    conn.execute("DELETE FROM materials")

# =================================================================
# PART A — materials
# =================================================================
print("\n[A] Loading materials from CSV...")

df = pd.read_csv(MATERIALS_CSV, dtype=str, keep_default_na=False)
df = df[df["material_id"].str.strip() != ""]

seen = set()
loaded = 0
for _, row in df.iterrows():
    raw = row.to_dict()
    mid = str(raw["material_id"]).strip()
    if mid in seen:
        print(f"    WARNING: duplicate material_id '{mid}' skipped")
        continue
    seen.add(mid)
    db.upsert_material(conn, db.row_to_sql(raw, provenance="curated"))
    loaded += 1

conn.commit()
print(f"    Inserted {loaded} materials from {MATERIALS_CSV}")

# Flag any class the app doesn't know — a typo here silently breaks family filters.
classes = {r["material_class"] for r in db.fetch_materials(conn)}
unknown = classes - db.KNOWN_CLASSES
if unknown:
    print(f"    WARNING: unexpected material_class values: {sorted(unknown)}")
print(f"    Material classes ({len(classes)}): {', '.join(sorted(classes))}")

# =================================================================
# PART B — allowable stress tables
# =================================================================
print("\n[B] Loading allowable-stress data...")

stress_df = pd.read_csv(STRESS_CSV)
conn.execute("DELETE FROM allowable_stress")      # fully derived from the CSV
for _, row in stress_df.iterrows():
    notes = row.get("notes", "")
    conn.execute("""
        INSERT INTO allowable_stress
        (material_id, stress_table_id, temperature_C, stress_MPa, source, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        row["material_id"],
        row["stress_table_id"],
        int(row["temperature_C"]),
        float(row["stress_MPa"]),
        row["source"],
        notes if pd.notna(notes) else "",
    ))
conn.commit()
print(f"    Inserted {len(stress_df)} stress rows")

# =================================================================
# SUMMARY
# =================================================================
print("\n" + "=" * 60)
print("Record store ready")
print(f"  materials:        {db.count_materials(conn)}")
print(f"  allowable_stress: {db.count_stress_rows(conn)}")
print(f"  file:             {db.DB_FILE}")
print("=" * 60)
print("\nNext: py reindex.py   (rebuilds the ChromaDB search index)")

conn.close()
