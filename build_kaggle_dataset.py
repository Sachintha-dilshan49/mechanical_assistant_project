"""
build_kaggle_dataset.py
Cleans the Kaggle "mechanical properties of materials" dump (data/Data.csv) and
maps it into our 43-column schema, writing data/kaggle_clean.csv.

This dataset is METALS ONLY and carries mechanical properties only. Every field
our recommendation engine normally reasons over (corrosion, chemical resistance,
weldability, cost, service temperature, applications, warnings, citations) is
absent, so those columns are written as NOT_FOUND. load_v1_data.py then merges
this file into the ChromaDB pool alongside the curated 66 materials.

Cleaning performed:
  - strip the " max" suffix on yield values (spec limits) -> numeric
  - drop rows with non-numeric / missing Su or Sy (unusable)
  - drop physically inconsistent rows where Su < Sy
  - convert E from MPa to GPa for elastic_modulus_GPa
  - best-effort material_class inference (documented as approximate)

NOTE: shear modulus G and Poisson's ratio mu are NOT in our schema, so the known
10x error in the dataset's stainless G column does not affect us (we don't import it).
"""

import csv
from pathlib import Path

RAW = "data/Data.csv"
OUT = "data/kaggle_clean.csv"
NOT_FOUND = "NOT_FOUND"

# Must match data/materials.csv exactly (column names + order).
COLUMNS = [
    "material_id", "common_name", "uns_number", "aisi_grade", "material_class",
    "condition", "yield_strength_MPa", "ultimate_tensile_strength_MPa",
    "elongation_percent", "hardness_HB", "hardness_shore_d", "fatigue_limit_MPa",
    "elastic_modulus_GPa", "density_kg_m3", "max_service_temp_C", "min_service_temp_C",
    "max_continuous_use_temp_C", "corrosion_seawater", "corrosion_acidic",
    "corrosion_alkaline", "corrosion_atmospheric", "corrosion_high_temp",
    "chemical_resistance_solvents", "chemical_resistance_acids",
    "chemical_resistance_alkalis", "chemical_resistance_fuels", "weldability",
    "weldability_notes", "joining_method", "machinability_index", "flammability",
    "water_absorption_percent", "uv_resistance", "cost_class", "approx_cost_usd_per_kg",
    "availability", "stock_forms", "fatigue_rating", "typical_applications",
    "key_warnings", "sources", "description_text", "stress_table_id",
]


def clean_num(v):
    """Float for a numeric cell; None if blank or non-numeric. Strips ' max'."""
    s = str(v).strip().replace(" max", "").replace("max", "").strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def num_or_nf(v):
    n = clean_num(v)
    if n is None:
        return NOT_FOUND
    # keep integers integer-looking
    return str(int(n)) if float(n).is_integer() else str(n)


def classify(material, desc, std):
    """Best-effort material_class using only our existing 24 classes.
    Approximate by design — the dataset has no composition fields."""
    s = (str(material) + " " + str(desc)).lower()

    if "cast iron" in s:
        return "cast_iron"
    if "magnesium" in s:
        return "magnesium_alloy"
    if "aluminum" in s or "aluminium" in s:
        return "aluminum_cast" if "cast" in s else "aluminum_wrought"
    if any(k in s for k in ("brass", "bronze", "copper", "nickel silver", "gunmetal")):
        return "copper_alloy"
    if "stainless" in s or "heat-resisting" in s or "corrosion-resistant" in s:
        # coarse: cannot reliably split austenitic/ferritic/martensitic here
        return "stainless_austenitic"

    # --- plain SAE/AISI grade parsing for ANSI steels ---
    import re
    m = re.search(r"\b([1-9]\d{3,4})\b", str(material))
    if m and "steel" in s:
        grade = m.group(1)
        if len(grade) == 5:           # 3xxxx / 5xxxx -> stainless families
            return "stainless_austenitic"
        first = grade[0]
        if first == "1":              # 10xx/11xx/12xx carbon; 13xx alloy
            if grade[:2] == "13":
                return "alloy_steel"
            cc = int(grade[2:4]) if grade[2:4].isdigit() else 30
            if cc <= 20:
                return "carbon_steel_low"
            if cc <= 50:
                return "carbon_steel_medium"
            return "carbon_steel_high"
        return "alloy_steel"          # 41xx/43xx/86xx/...

    # --- description-based fallback for EN/DIN/BS/GOST/JIS/NF/CSN ---
    if "alloy" in s or "spring" in s or "nitrided" in s or "heat resistant" in s \
            or "heat-resistant" in s:
        return "alloy_steel"
    if "cast steel" in s:
        return "alloy_steel" if "alloy" in s else "carbon_steel_medium"
    if "steel" in s or "iron" in s:
        return "carbon_steel_medium"
    return "carbon_steel_medium"


def main():
    raw_path = Path(RAW)
    if not raw_path.exists():
        raise SystemExit(
            f"ERROR: {RAW} not found. Save the Kaggle CSV there first "
            f"(columns: Std,ID,Material,Heat treatment,Su,Sy,A5,Bhn,E,G,mu,Ro,pH,Desc,HV)."
        )

    with open(raw_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"Read {len(rows)} raw rows from {RAW}")

    out_rows = []
    seen_ids = set()
    skipped_badnum = 0
    skipped_inconsistent = 0
    skipped_dupe = 0
    class_counts = {}

    for r in rows:
        su = clean_num(r.get("Su"))
        sy = clean_num(r.get("Sy"))
        if su is None or sy is None:
            skipped_badnum += 1
            continue
        if su < sy:                       # UTS must exceed yield for a real pair
            skipped_inconsistent += 1
            continue

        rid = "kag_" + str(r.get("ID", "")).strip()[:12].lower()
        if not rid.strip("kag_") or rid in seen_ids:
            skipped_dupe += 1
            continue
        seen_ids.add(rid)

        material = str(r.get("Material", "")).strip()
        heat = str(r.get("Heat treatment", "")).strip()
        std = str(r.get("Std", "")).strip()
        desc = str(r.get("Desc", "")).strip()
        cls = classify(material, desc, std)
        class_counts[cls] = class_counts.get(cls, 0) + 1

        e_gpa = clean_num(r.get("E"))
        e_gpa = str(round(e_gpa / 1000.0, 1)) if e_gpa else NOT_FOUND
        ro = num_or_nf(r.get("Ro"))
        a5 = num_or_nf(r.get("A5"))
        bhn = num_or_nf(r.get("Bhn"))

        common = material if material else f"{std} material"

        # Honest, factual description for semantic embedding.
        parts = [f"{common} is a {cls.replace('_', ' ')} ({std} standard)."]
        parts.append(f"Ultimate tensile strength {num_or_nf(su)} MPa, "
                     f"yield strength {num_or_nf(sy)} MPa.")
        extra = []
        if a5 != NOT_FOUND:
            extra.append(f"elongation {a5}%")
        if bhn != NOT_FOUND:
            extra.append(f"Brinell hardness {bhn} HB")
        if e_gpa != NOT_FOUND:
            extra.append(f"elastic modulus {e_gpa} GPa")
        if ro != NOT_FOUND:
            extra.append(f"density {ro} kg/m3")
        if extra:
            parts.append("Also " + ", ".join(extra) + ".")
        if heat:
            parts.append(f"Condition: {heat}.")
        parts.append("Mechanical property data only — no corrosion, cost, or "
                     "service-temperature data is available for this entry.")
        description = " ".join(parts)

        row = {c: NOT_FOUND for c in COLUMNS}
        row.update({
            "material_id": rid,
            "common_name": common,
            "material_class": cls,
            "condition": (heat.replace(" ", "_") if heat else "as_supplied"),
            "yield_strength_MPa": num_or_nf(sy),
            "ultimate_tensile_strength_MPa": num_or_nf(su),
            "elongation_percent": a5,
            "hardness_HB": bhn,
            "elastic_modulus_GPa": e_gpa,
            "density_kg_m3": ro,
            "sources": f"Kaggle mechanical properties dataset ({std})",
            "description_text": description,
            "stress_table_id": "",     # empty -> stress lookup skips gracefully
        })
        out_rows.append(row)

    Path("data").mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} cleaned rows to {OUT}")
    print(f"  skipped (bad/missing Su or Sy): {skipped_badnum}")
    print(f"  skipped (Su < Sy inconsistent): {skipped_inconsistent}")
    print(f"  skipped (dupe/blank id):        {skipped_dupe}")
    print("\nInferred material_class distribution (approximate):")
    for k, v in sorted(class_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {v:>5}  {k}")
    print("\nNext: run  python load_v1_data.py")


if __name__ == "__main__":
    main()
