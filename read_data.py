# read_data.py
# Reads the materials spreadsheet and prints what's in it

import pandas as pd

# Path to our spreadsheet (forward slashes work fine on Windows in Python)
EXCEL_FILE = "data/materials_database_template.xlsx"

# Read the "Materials" sheet from the Excel file
materials_df = pd.read_excel(EXCEL_FILE, sheet_name="Materials")

# Drop rows where material_id is empty (these are blank rows in the spreadsheet)
materials_df = materials_df.dropna(subset=['material_id'])

# Print a summary so we can see what loaded
print("=" * 60)
print(f"Loaded {len(materials_df)} rows from the Materials sheet")
print("=" * 60)

# Show the first row (SS 316L) with just the key columns
print("\nFirst material in the spreadsheet:")
print("-" * 60)

# Loop through every material in the spreadsheet
for index, row in materials_df.iterrows():
    print(f"\nMaterial #{index + 1}:")
    print(f"  Material ID:    {row['material_id']}")
    print(f"  Common name:    {row['common_name']}")
    print(f"  Class:          {row['material_class']}")
    print(f"  Yield strength: {row['yield_strength_MPa']} MPa")
    print(f"  Max temp:       {row['max_service_temp_C']} °C")
    print(f"  Weldability:    {row['weldability']}/5")