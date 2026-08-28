"""
migrate_add_confidence.py
Adds the data_confidence column to an existing data/materials.db.

init_db.py refuses to overwrite a populated table (and create_schema only ever
runs CREATE TABLE IF NOT EXISTS), so a database that was seeded before this
column existed will never grow it on its own. This script is the only supported
way to bring a live database up to the current schema.

Every existing row is backfilled with "estimated": the rows predate the column,
so nothing is actually known about how they were sourced, and under-claiming is
the safe direction. Upgrade rows to cross_referenced / verified_primary_source
by hand once each one has been checked.

Safe to run twice - if the column is already there it changes nothing and exits.

USAGE:
    py migrate_add_confidence.py
"""

import sys

import db

COLUMN = "data_confidence"

print("=" * 60)
print("Migration: add data_confidence to the materials table")
print("=" * 60)

conn = db.connect()

# The materials table may not exist at all (fresh clone, never seeded). Say so
# rather than letting sqlite raise from the ALTER below.
table = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name='materials'"
).fetchone()
if table is None:
    print(f"\n    No materials table in {db.DB_FILE}.")
    print("    Nothing to migrate - run: py init_db.py")
    conn.close()
    sys.exit(0)

existing = {r["name"] for r in conn.execute("PRAGMA table_info(materials)")}
if COLUMN in existing:
    print(f"\n    Column '{COLUMN}' is already present - nothing to do.")
    counts = conn.execute(
        f"SELECT {COLUMN}, COUNT(*) AS n FROM materials GROUP BY {COLUMN}"
    ).fetchall()
    for row in counts:
        print(f"      {row[COLUMN] if row[COLUMN] is not None else 'NULL':<24} {row['n']}")
    conn.close()
    sys.exit(0)

rows = db.count_materials(conn)
print(f"\n[1/2] Adding column '{COLUMN}' TEXT ({rows} rows in the table)")
conn.execute(f"ALTER TABLE materials ADD COLUMN {COLUMN} TEXT")

# Backfill in one statement, including the rows ALTER TABLE just filled with
# NULL. The WHERE keeps a re-run (or a partially populated table) from
# overwriting a confidence that has already been upgraded by hand.
print(f"[2/2] Backfilling every row with '{db.DEFAULT_DATA_CONFIDENCE}'")
cur = conn.execute(
    f"UPDATE materials SET {COLUMN} = ? WHERE {COLUMN} IS NULL OR TRIM({COLUMN}) = ''",
    (db.DEFAULT_DATA_CONFIDENCE,),
)
conn.commit()
print(f"      {cur.rowcount} rows backfilled")

counts = conn.execute(
    f"SELECT {COLUMN}, COUNT(*) AS n FROM materials GROUP BY {COLUMN}"
).fetchall()
print("\n" + "=" * 60)
print("Migration complete")
for row in counts:
    print(f"  {row[COLUMN]:<24} {row['n']}")
print("=" * 60)
print("\nNext: py reindex.py   (so the field reaches the search index)")

conn.close()
