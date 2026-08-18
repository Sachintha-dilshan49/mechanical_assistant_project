"""
db.py
The materials record store — SQLite is the source of truth for material data.

ChromaDB is a derived search index, rebuilt from this table by reindex.py, and is
safe to delete at any time. This file is the one place that knows the schema, so
the loader, the reindexer and the future PDF ingester all agree on it.

Missing values are NULL here (so SQL range filters behave correctly). They are
converted to the "NOT_FOUND" sentinel only on the way out to ChromaDB, because
Chroma metadata cannot hold nulls and the rest of the app already reads that
sentinel as "not applicable".
"""

import sqlite3
from datetime import datetime, timezone

DB_FILE = "data/materials.db"

# Sentinel the app layer uses for a property that does not apply to a material.
NOT_FOUND = "NOT_FOUND"

# -----------------------------------------------------------------
# SCHEMA — (column, sql_type). Order matches data/materials.csv so the two
# stay easy to compare by eye.
#   TEXT    identity, notes, pipe-separated lists, prose
#   REAL    measured physical quantities
#   INTEGER 1-5 subjective ratings (constrained below)
# -----------------------------------------------------------------
MATERIAL_COLUMNS = [
    # identity / classification
    ("material_id",                   "TEXT"),
    ("common_name",                   "TEXT"),
    ("uns_number",                    "TEXT"),
    ("aisi_grade",                    "TEXT"),
    ("material_class",                "TEXT"),
    ("condition",                     "TEXT"),
    # mechanical
    ("yield_strength_MPa",            "REAL"),
    ("ultimate_tensile_strength_MPa", "REAL"),
    ("elongation_percent",            "REAL"),
    ("hardness_HB",                   "REAL"),
    ("hardness_shore_d",              "REAL"),
    ("fatigue_limit_MPa",             "REAL"),
    ("elastic_modulus_GPa",           "REAL"),
    ("density_kg_m3",                 "REAL"),
    # thermal
    ("max_service_temp_C",            "REAL"),
    ("min_service_temp_C",            "REAL"),
    ("max_continuous_use_temp_C",     "REAL"),
    # corrosion (metals, 1-5)
    ("corrosion_seawater",            "INTEGER"),
    ("corrosion_acidic",              "INTEGER"),
    ("corrosion_alkaline",            "INTEGER"),
    ("corrosion_atmospheric",         "INTEGER"),
    ("corrosion_high_temp",           "INTEGER"),
    # chemical resistance (non-metals, 1-5)
    ("chemical_resistance_solvents",  "INTEGER"),
    ("chemical_resistance_acids",     "INTEGER"),
    ("chemical_resistance_alkalis",   "INTEGER"),
    ("chemical_resistance_fuels",     "INTEGER"),
    # joining / fabrication
    ("weldability",                   "INTEGER"),
    ("weldability_notes",             "TEXT"),
    ("joining_method",                "TEXT"),
    ("machinability_index",           "REAL"),
    # environmental / safety
    ("flammability",                  "TEXT"),
    ("water_absorption_percent",      "REAL"),
    ("uv_resistance",                 "INTEGER"),
    # commercial
    ("cost_class",                    "INTEGER"),
    ("approx_cost_usd_per_kg",        "TEXT"),   # ranges like "4-7" are stored verbatim
    ("availability",                  "INTEGER"),
    ("stock_forms",                   "TEXT"),
    ("fatigue_rating",                "INTEGER"),
    # text / provenance
    ("typical_applications",          "TEXT"),
    ("key_warnings",                  "TEXT"),
    ("sources",                       "TEXT"),
    ("description_text",              "TEXT"),
    ("stress_table_id",               "TEXT"),
]

COLUMN_NAMES = [c for c, _ in MATERIAL_COLUMNS]

# Bookkeeping columns the CSV does not have. They exist so a material added later
# from a PDF can be told apart from the hand-curated seed set, and rolled back.
EXTRA_COLUMNS = [
    ("provenance", "TEXT"),   # 'curated' | 'extracted'
    ("added_at",   "TEXT"),   # ISO-8601 UTC
]

# Ratings are 1-5 by definition; a bad extraction should fail loudly at write time
# rather than quietly poison the recommendations.
RATING_COLUMNS = [c for c, t in MATERIAL_COLUMNS
                  if t == "INTEGER" and c not in ("machinability_index",)]

# Every material_class the app knows about. Kept here so the loader, the
# reindexer and the ingester validate against one list.
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


# =================================================================
# CONNECTION
# =================================================================
def connect(db_file=DB_FILE):
    """Open the database with dict-style row access and FK enforcement on."""
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# =================================================================
# SCHEMA CREATION
# =================================================================
def create_schema(conn):
    """Create the materials and allowable_stress tables if they don't exist.

    Never drops anything — seeding and reindexing are separate, explicit steps,
    so a stray run of a build script can't wipe materials you added yourself.
    """
    cols = []
    for name, sql_type in MATERIAL_COLUMNS + EXTRA_COLUMNS:
        if name == "material_id":
            cols.append(f"{name} {sql_type} PRIMARY KEY")
        elif name in ("common_name", "material_class", "description_text"):
            cols.append(f"{name} {sql_type} NOT NULL")
        else:
            cols.append(f"{name} {sql_type}")

    # NULL is always allowed — it means "property does not apply to this material".
    checks = [f"CHECK ({c} IS NULL OR {c} BETWEEN 1 AND 5)" for c in RATING_COLUMNS]
    checks.append("CHECK (density_kg_m3 IS NULL OR density_kg_m3 > 0)")

    # Built outside the f-string: Python 3.11 rejects backslashes in f-string
    # expressions, and this project runs on 3.11.
    body = ",\n            ".join(cols + checks)
    conn.execute(f"CREATE TABLE IF NOT EXISTS materials (\n            {body}\n        )")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_materials_class ON materials(material_class)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS allowable_stress (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            material_id     TEXT NOT NULL,
            stress_table_id TEXT NOT NULL,
            temperature_C   INTEGER NOT NULL,
            stress_MPa      REAL NOT NULL,
            source          TEXT NOT NULL,
            notes           TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_stress_lookup
        ON allowable_stress(stress_table_id, temperature_C)
    """)
    conn.commit()


# =================================================================
# VALUE COERCION  (CSV/LLM text  ->  SQL value)
# =================================================================
def to_number(val):
    """Float for a real value, None for blank / NOT_FOUND / unparseable."""
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.upper() == NOT_FOUND:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def to_int(val):
    """Rounded int, or None. Used for the 1-5 rating columns."""
    n = to_number(val)
    return int(round(n)) if n is not None else None


def to_text(val):
    """Trimmed string, or None for blank / NOT_FOUND."""
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.upper() == NOT_FOUND:
        return None
    return s


def row_to_sql(row, provenance="curated"):
    """Map one raw row (CSV dict or extractor output) to SQL values.

    Applies the right coercion per column type, so 'NOT_FOUND' and '' both land
    as NULL and range filters behave.
    """
    out = {}
    for name, sql_type in MATERIAL_COLUMNS:
        raw = row.get(name)
        if sql_type == "REAL":
            out[name] = to_number(raw)
        elif sql_type == "INTEGER":
            out[name] = to_int(raw)
        else:
            out[name] = to_text(raw)
    out["provenance"] = provenance
    out["added_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out


# =================================================================
# WRITES
# =================================================================
def upsert_material(conn, values):
    """Insert a material, replacing any existing row with the same material_id."""
    all_cols = COLUMN_NAMES + [c for c, _ in EXTRA_COLUMNS]
    placeholders = ", ".join("?" for _ in all_cols)
    conn.execute(
        f"INSERT OR REPLACE INTO materials ({', '.join(all_cols)}) VALUES ({placeholders})",
        [values.get(c) for c in all_cols],
    )


# =================================================================
# READS
# =================================================================
def fetch_materials(conn):
    """Every material as a plain dict, missing values as None."""
    rows = conn.execute(
        f"SELECT * FROM materials ORDER BY material_id"
    ).fetchall()
    return [dict(r) for r in rows]


def count_materials(conn):
    return conn.execute("SELECT COUNT(*) FROM materials").fetchone()[0]


def count_stress_rows(conn):
    return conn.execute("SELECT COUNT(*) FROM allowable_stress").fetchone()[0]


# =================================================================
# SQL  ->  ChromaDB metadata
# =================================================================
def to_chroma_metadata(material):
    """Convert a material row to ChromaDB metadata.

    Chroma metadata values cannot be None, so NULL becomes the "NOT_FOUND"
    sentinel string — which is exactly what retrieve.py, reason.py and app.py
    already expect for a property that doesn't apply. Numbers stay floats so
    $gte/$lte range filters keep working.
    """
    meta = {}
    for name, sql_type in MATERIAL_COLUMNS:
        if name == "material_id":
            continue                      # carried as the Chroma id, not metadata
        value = material.get(name)
        if value is None:
            meta[name] = NOT_FOUND
        elif sql_type in ("REAL", "INTEGER"):
            meta[name] = float(value)
        else:
            meta[name] = str(value)
    return meta
