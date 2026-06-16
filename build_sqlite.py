# build_sqlite.py
# Reads the Stress_Temperature sheet and loads it into a SQLite database

import pandas as pd
import sqlite3
from pathlib import Path

# File paths
EXCEL_FILE = "data/materials_database_template.xlsx"
DB_FILE = "data/materials.db"

print("=" * 60)
print("Building SQLite database from spreadsheet")
print("=" * 60)

# -----------------------------------------------------------------
# STEP 1: Read the spreadsheet
# -----------------------------------------------------------------
print("\n[1/4] Reading Stress_Temperature sheet from Excel...")

stress_df = pd.read_excel(EXCEL_FILE, sheet_name="Stress_Temperature")

# Drop empty rows
stress_df = stress_df.dropna(subset=['material_id'])

print(f"      Loaded {len(stress_df)} stress-temperature rows")

# -----------------------------------------------------------------
# STEP 2: Connect to SQLite (creates the file if it doesn't exist)
# -----------------------------------------------------------------
print(f"\n[2/4] Connecting to database: {DB_FILE}")

# Make sure the data folder exists
Path("data").mkdir(exist_ok=True)

# Connect (creates materials.db if not present)
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# -----------------------------------------------------------------
# STEP 3: Create the table
# -----------------------------------------------------------------
print("\n[3/4] Creating table 'allowable_stress'...")

# Drop the table if it already exists, so we start fresh each time
cursor.execute("DROP TABLE IF EXISTS allowable_stress")

# Create the table with the columns we need
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

# Create an index to make lookups fast
cursor.execute("""
    CREATE INDEX idx_stress_lookup 
    ON allowable_stress(stress_table_id, temperature_C)
""")

print("      Table and index created")

# -----------------------------------------------------------------
# STEP 4: Insert the data from the spreadsheet
# -----------------------------------------------------------------
print("\n[4/4] Inserting data...")

inserted_count = 0
for index, row in stress_df.iterrows():
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
    inserted_count += 1

# Save the changes (this writes to the file)
conn.commit()

print(f"      Inserted {inserted_count} rows")

# -----------------------------------------------------------------
# VERIFICATION: Read back and show what we stored
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("Verification: reading data back from database")
print("=" * 60)

cursor.execute("""
    SELECT material_id, temperature_C, stress_MPa, source
    FROM allowable_stress
    ORDER BY material_id, temperature_C
""")

results = cursor.fetchall()

print(f"\nFound {len(results)} rows in the database:")
print("-" * 60)
print(f"{'Material':<25} {'Temp (°C)':<12} {'Stress (MPa)':<15}")
print("-" * 60)

for material_id, temp, stress, source in results:
    print(f"{material_id:<25} {temp:<12} {stress:<15}")

# Close the connection
conn.close()

print("\n" + "=" * 60)
print("Database built successfully")
print(f"Saved to: {DB_FILE}")
print("=" * 60)