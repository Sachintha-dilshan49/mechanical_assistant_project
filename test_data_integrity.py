"""
test_data_integrity.py
Data and input checks that need no API calls: malformed input, SQL injection,
stress-table boundaries, physical plausibility, SQLite/ChromaDB agreement.

    py test_data_integrity.py
"""
import sys, sqlite3, time, warnings
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FAILS = []


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append((name, detail))


print("=" * 74)
print("A. INPUT EDGE CASES  (classify_local / retrieval must not crash)")
print("=" * 74)
from reason import classify_local, _normalise
from retrieve import find_candidates, find_materials, get_allowable_stress

EDGE = {
    "empty": "",
    "spaces only": "     ",
    "newlines": "\n\n\n",
    "single char": "a",
    "punctuation only": "?!?!...",
    "emoji only": "🔧🔧🔧",
    "unicode": "материал для лодки",
    "very long": "steel " * 2000,
    "sql-ish": "'; DROP TABLE materials; --",
    "html": "<script>alert(1)</script>",
    "null byte": "steel\x00bracket",
    "control chars": "steel\x01\x02bracket",
}
for label, q in EDGE.items():
    try:
        classify_local(q)
        check(f"classify_local: {label}", True)
    except Exception as exc:
        check(f"classify_local: {label}", False, f"{type(exc).__name__}: {exc}")

for label, q in EDGE.items():
    try:
        pool = find_candidates(q, pool_size=5)
        check(f"retrieval: {label}", True, f"{len(pool)} results")
    except Exception as exc:
        check(f"retrieval: {label}", False, f"{type(exc).__name__}: {str(exc)[:90]}")

print()
print("=" * 74)
print("B. SQL INJECTION into the stress lookup")
print("=" * 74)
try:
    r = get_allowable_stress("316L_annealed'; DROP TABLE allowable_stress; --", 200)
    conn = sqlite3.connect("data/materials.db")
    still = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name='allowable_stress'").fetchone()[0]
    conn.close()
    check("table survives injection attempt", still == 1, f"returned {r}")
except Exception as exc:
    check("injection attempt handled", False, str(exc)[:90])

print()
print("=" * 74)
print("C. STRESS LOOKUP boundaries and correctness")
print("=" * 74)
lo = get_allowable_stress("316L_annealed", -273)
hi = get_allowable_stress("316L_annealed", 99999)
check("below range returns error, not a value",
      bool(lo and lo.get("stress_MPa") is None), str(lo)[:70])
check("above range returns error, not a value",
      bool(hi and hi.get("stress_MPa") is None), str(hi)[:70])

conn = sqlite3.connect("data/materials.db")
rows = conn.execute("SELECT temperature_C, stress_MPa FROM allowable_stress "
                    "WHERE stress_table_id='316L_annealed' ORDER BY temperature_C").fetchall()
conn.close()
t_lo, s_lo = rows[0]
t_hi, s_hi = rows[-1]
edge_lo = get_allowable_stress("316L_annealed", t_lo)
edge_hi = get_allowable_stress("316L_annealed", t_hi)
check("exact lowest point", bool(edge_lo) and edge_lo.get("stress_MPa") == s_lo,
      f"{t_lo}C -> {edge_lo.get('stress_MPa') if edge_lo else None} (expect {s_lo})")
check("exact highest point", bool(edge_hi) and edge_hi.get("stress_MPa") == s_hi,
      f"{t_hi}C -> {edge_hi.get('stress_MPa') if edge_hi else None} (expect {s_hi})")

# midpoint interpolation should sit between its two neighbours
(t1, s1), (t2, s2) = rows[0], rows[1]
mid = get_allowable_stress("316L_annealed", (t1 + t2) / 2)
expect = s1 + (s2 - s1) * 0.5
check("midpoint interpolation is linear",
      bool(mid) and abs(mid["stress_MPa"] - expect) < 0.06,
      f"got {mid.get('stress_MPa') if mid else None}, expect ~{expect:.1f}")

check("unknown table id returns None", get_allowable_stress("does_not_exist", 200) is None)
try:
    nan_res = get_allowable_stress("316L_annealed", float("nan"))
    check("NaN temperature handled", True, f"returned {str(nan_res)[:60]}")
except Exception as exc:
    check("NaN temperature handled", False, f"{type(exc).__name__}: {exc}")

print()
print("=" * 74)
print("D. DATABASE INTEGRITY")
print("=" * 74)
import db
conn = db.connect()
mats = db.fetch_materials(conn)
ids = [m["material_id"] for m in mats]
check("no duplicate material_id", len(ids) == len(set(ids)), f"{len(ids)} rows")
check("every row has description_text",
      all((m.get("description_text") or "").strip() for m in mats))
bad_class = [m["material_id"] for m in mats if m["material_class"] not in db.KNOWN_CLASSES]
check("all material_class values known", not bad_class, str(bad_class[:4]))
# yield must not exceed UTS where both are present
# Metals only. Ductile polymers legitimately yield then neck and draw, so their
# stress at break is LOWER than the yield peak (nylon 83/82, PP 35/33, PVC 55/52
# are all correct). Applying a metals rule to them reports false failures.
bad_strength = [m["material_id"] for m in mats
                if not str(m.get("material_class", "")).startswith(("plastic_", "composite_"))
                and m.get("yield_strength_MPa") and m.get("ultimate_tensile_strength_MPa")
                and m["yield_strength_MPa"] > m["ultimate_tensile_strength_MPa"]]
check("metals: yield <= UTS", not bad_strength, str(bad_strength[:5]))
bad_rho = [m["material_id"] for m in mats
           if m.get("density_kg_m3") is not None and not (100 < m["density_kg_m3"] < 25000)]
check("densities physically plausible", not bad_rho, str(bad_rho[:5]))
bad_temp = [m["material_id"] for m in mats
            if m.get("min_service_temp_C") is not None
            and m.get("max_service_temp_C") is not None
            and m["min_service_temp_C"] >= m["max_service_temp_C"]]
check("min_service_temp < max_service_temp", not bad_temp, str(bad_temp[:5]))
# every stress_table_id referenced by a material should exist in the stress table
stress_ids = {r[0] for r in conn.execute(
    "SELECT DISTINCT stress_table_id FROM allowable_stress")}
dangling = [m["material_id"] for m in mats
            if m.get("stress_table_id") and m["stress_table_id"] not in stress_ids]
check("no dangling stress_table_id references", not dangling, str(dangling[:5]))
conn.close()

# Chroma and SQLite must agree on what exists
import chromadb
col = chromadb.PersistentClient(path="data/chroma_db").get_collection("materials")
chroma_ids = set(col.get()["ids"])
check("ChromaDB and SQLite hold the same ids",
      chroma_ids == set(ids),
      f"chroma={len(chroma_ids)} sqlite={len(ids)} "
      f"only_in_chroma={list(chroma_ids - set(ids))[:3]} "
      f"only_in_sqlite={list(set(ids) - chroma_ids)[:3]}")

print()
print("=" * 74)
print("E. RESOURCE HANDLING")
print("=" * 74)
warnings.simplefilter("error", ResourceWarning)
try:
    for _ in range(60):
        get_allowable_stress("316L_annealed", 200)
    check("60 stress lookups leak no connections", True)
except ResourceWarning as exc:
    check("60 stress lookups leak no connections", False, str(exc)[:90])
warnings.resetwarnings()

import llm
t0 = time.time()
for _ in range(40):
    llm.usage_since(24)
check("40 usage reads complete quickly", time.time() - t0 < 6,
      f"{time.time() - t0:.2f}s")

print()
print("=" * 74)
print(f"RESULT: {len(FAILS)} failure(s)")
for name, detail in FAILS:
    print(f"  - {name}: {detail}")
print("=" * 74)
