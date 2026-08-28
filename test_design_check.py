"""
test_design_check.py
Verification for design_check.py, against closed-form answers.

Unlike retrieval — where "correct" is a judgement call and the benchmark can
only measure agreement with hand-picked answers — mechanics has exact answers.
Every expected value below is computed from the textbook formula independently
of the module, so a wrong constant in BEAMS or a slipped unit conversion fails
here rather than reaching a student as a confident factor of safety.

No API calls, no quota. Run any time:

    py test_design_check.py
"""
import sys, math

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import design_check as dc

FAILS = []


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append((name, detail))


def close(name, got, want, tol=1e-6):
    """Compare against a hand-computed value with a relative tolerance."""
    if got is None:
        check(name, False, f"got None, expected {want:g}")
        return
    rel = abs(got - want) / abs(want) if want else abs(got)
    check(name, rel < tol, f"got {got:.6g}, expected {want:.6g}")


def raises(name, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as exc:
        check(name, True, str(exc)[:70])
    except Exception as exc:
        check(name, False, f"wrong exception {type(exc).__name__}: {exc}")
    else:
        check(name, False, "no exception raised")


# Test materials. Deliberately synthetic and round-numbered so the expected
# values below are arithmetic anyone can repeat by hand.
STEEL = {
    "material_id": "TEST_STEEL",
    "common_name": "Test Steel",
    "material_class": "carbon_steel_medium",
    "yield_strength_MPa": 250.0,
    "ultimate_tensile_strength_MPa": 400.0,
    "elastic_modulus_GPa": 200.0,
    "fatigue_limit_MPa": 180.0,
    "max_service_temp_C": 400.0,
    "min_service_temp_C": -30.0,
}
CERAMIC = {                      # brittle: no yield point, as in the real rows
    "material_id": "TEST_CERAMIC",
    "common_name": "Test Ceramic",
    "material_class": "ceramic_oxide",
    "yield_strength_MPa": None,
    "ultimate_tensile_strength_MPa": 300.0,
    "elastic_modulus_GPa": 370.0,
    "fatigue_limit_MPa": None,
    "max_service_temp_C": 1500.0,
}
FROM_CHROMA = {                  # the sentinel form, as it comes back from Chroma
    "material_id": "TEST_SENTINEL",
    "common_name": "Test Sentinel",
    "material_class": "plastic_thermoplastic",
    "yield_strength_MPa": 60.0,
    "ultimate_tensile_strength_MPa": 70.0,
    "elastic_modulus_GPa": 2.8,
    "fatigue_limit_MPa": "NOT_FOUND",
    "max_service_temp_C": 90.0,
    "min_service_temp_C": "NOT_FOUND",
}

print("=" * 74)
print("A. SECTION PROPERTIES  (exact closed-form)")
print("=" * 74)

r20 = dc.solid_round(20)
close("solid round d=20: A = pi*d^2/4", r20["A_mm2"], math.pi * 400 / 4)
close("solid round d=20: I = pi*d^4/64", r20["I_mm4"], math.pi * 160000 / 64)
close("solid round d=20: c = d/2", r20["c_mm"], 10.0)
close("solid round d=20: J = 2I", r20["J_mm4"], 2 * r20["I_mm4"])

t = dc.tube(20, 16)
close("tube 20x16: A", t["A_mm2"], math.pi * (400 - 256) / 4)
close("tube 20x16: I", t["I_mm4"], math.pi * (160000 - 65536) / 64)
close("tube 20x16: J = 2I", t["J_mm4"], 2 * t["I_mm4"])
check("tube I < solid I at the same OD", t["I_mm4"] < r20["I_mm4"],
      f"{t['I_mm4']:.1f} < {r20['I_mm4']:.1f}")

rect = dc.rectangle(10, 20)
close("rectangle 10x20: A = b*h", rect["A_mm2"], 200.0)
close("rectangle 10x20: I = b*h^3/12", rect["I_mm4"], 10 * 8000 / 12)
close("rectangle 10x20: c = h/2", rect["c_mm"], 10.0)
check("rectangle J is None (needs the series solution, not I)",
      rect["J_mm4"] is None)

# A rectangle on edge is far stiffer than the same one flat: I scales with h^3,
# so swapping 10x20 for 20x10 must change I by a factor of exactly 4.
close("rectangle I scales with h^3 (20x10 vs 10x20)",
      dc.rectangle(20, 10)["I_mm4"], rect["I_mm4"] / 4)

print()
print("=" * 74)
print("B. GEOMETRY VALIDATION  (bad input must fail at entry)")
print("=" * 74)
raises("negative diameter rejected", dc.solid_round, -20)
raises("zero diameter rejected", dc.solid_round, 0)
raises("non-numeric diameter rejected", dc.solid_round, "twenty")
raises("NaN diameter rejected", dc.solid_round, float("nan"))
raises("infinite diameter rejected", dc.solid_round, float("inf"))
raises("tube bore >= outer diameter rejected", dc.tube, 20, 20)
raises("tube bore larger than outer diameter rejected", dc.tube, 20, 25)
check("tube with zero bore is a solid bar, not an error",
      abs(dc.tube(20, 0)["I_mm4"] - r20["I_mm4"]) < 1e-9)

print()
print("=" * 74)
print("C. AXIAL:  sigma = F/A")
print("=" * 74)

# 10 kN through a 20 mm bar: 10000 / 314.159 = 31.831 MPa
ax = dc.check_axial(STEEL, r20, 10000)
close("stress = F/A", ax["stress_MPa"], 10000 / (math.pi * 100))
close("FoS = Sy/sigma", ax["fos"], 250 / (10000 / (math.pi * 100)))
check("basis is yield for a ductile metal", ax["basis"] == "yield", ax["basis"])
check("verdict pass at FoS 7.9", ax["verdict"] == "pass", ax["verdict"])
check("Kt caveat is stated", any("Kt" in a for a in ax["assumptions"]))
check("no deflection for an axial check", ax["deflection_mm"] is None)

# Stress must scale as 1/d^2 in axial loading.
close("axial stress scales 1/d^2 (d=40 vs d=20)",
      dc.check_axial(STEEL, dc.solid_round(40), 10000)["stress_MPa"],
      ax["stress_MPa"] / 4)

# 100 kN through the same bar: 318.3 MPa, past yield and past ultimate.
over = dc.check_axial(STEEL, r20, 100000)
check("verdict fail when stress exceeds yield", over["verdict"] == "fail",
      f"FoS {over['fos']:.2f}")
check("FoS below 1 when past yield", over["fos"] < 1.0)
check("ultimate FoS reported alongside yield FoS",
      over["fos_ultimate"] > over["fos_yield"])

# The marginal band sits between 1.0 and the target.
marg = dc.check_axial(STEEL, r20, 65000)      # 206.9 MPa -> FoS 1.21
check("verdict marginal between FoS 1 and target",
      marg["verdict"] == "marginal", f"FoS {marg['fos']:.2f}")
check("target FoS is configurable",
      dc.check_axial(STEEL, r20, 65000, target_fos=1.2)["verdict"] == "pass")

check("tension and compression give the same magnitude of stress",
      abs(dc.check_axial(STEEL, r20, -10000)["stress_MPa"] - ax["stress_MPa"]) < 1e-9)
check("compression warns that buckling is not checked",
      any("buckl" in w.lower() for w in dc.check_axial(STEEL, r20, -10000)["warnings"]))

raises("non-numeric force rejected", dc.check_axial, STEEL, r20, "heavy")
raises("NaN force rejected", dc.check_axial, STEEL, r20, float("nan"))

print()
print("=" * 74)
print("D. BENDING:  sigma = M*c/I,  with matching deflection")
print("=" * 74)

I20 = math.pi * 160000 / 64          # 7853.98 mm^4
E = 200_000.0                        # 200 GPa in MPa
EI = E * I20                         # 1.5708e9 N*mm^2

# --- cantilever, 1 kN at the free end, 500 mm ---
# M = P*L = 500,000 N*mm ; sigma = M*c/I = 636.62 MPa ; d = PL^3/3EI = 26.526 mm
b1 = dc.check_bending(STEEL, r20, 1000, 500, "cantilever_end")
close("cantilever end load: M = P*L", b1["load"]["moment_Nmm"], 500_000)
close("cantilever end load: sigma = M*c/I", b1["stress_MPa"], 500_000 * 10 / I20)
close("cantilever end load: d = P*L^3/(3EI)", b1["deflection_mm"],
      1000 * 500 ** 3 / (3 * EI))
close("modulus converted GPa -> MPa", b1["modulus_MPa"], 200_000)

# --- simply supported, 1 kN at mid-span, 500 mm ---
# M = P*L/4 = 125,000 ; d = PL^3/48EI = 1.6579 mm
b2 = dc.check_bending(STEEL, r20, 1000, 500, "simple_center")
close("simple centre load: M = P*L/4", b2["load"]["moment_Nmm"], 125_000)
close("simple centre load: sigma", b2["stress_MPa"], 125_000 * 10 / I20)
close("simple centre load: d = P*L^3/(48EI)", b2["deflection_mm"],
      1000 * 500 ** 3 / (48 * EI))
check("a cantilever bends 16x more than the same simply supported beam",
      abs(b1["deflection_mm"] / b2["deflection_mm"] - 16) < 1e-9,
      f"ratio {b1['deflection_mm'] / b2['deflection_mm']:.4f}")

# --- cantilever, 2 N/mm UDL over 1000 mm ---
# M = w*L^2/2 = 1,000,000 ; d = wL^4/8EI = 159.15 mm
b3 = dc.check_bending(STEEL, r20, 2, 1000, "cantilever_udl")
close("cantilever UDL: M = w*L^2/2", b3["load"]["moment_Nmm"], 1_000_000)
close("cantilever UDL: d = w*L^4/(8EI)", b3["deflection_mm"],
      2 * 1000 ** 4 / (8 * EI))

# --- simply supported, 2 N/mm UDL over 1000 mm ---
# M = w*L^2/8 = 250,000 ; d = 5wL^4/384EI = 16.579 mm
b4 = dc.check_bending(STEEL, r20, 2, 1000, "simple_udl")
close("simple UDL: M = w*L^2/8", b4["load"]["moment_Nmm"], 250_000)
close("simple UDL: d = 5w*L^4/(384EI)", b4["deflection_mm"],
      5 * 2 * 1000 ** 4 / (384 * EI))

check("point-load beams are labelled N", b1["load"]["unit"] == "N")
check("distributed-load beams are labelled N/mm", b4["load"]["unit"] == "N/mm")

# Bending stress must scale as 1/d^3.
close("bending stress scales 1/d^3 (d=40 vs d=20)",
      dc.check_bending(STEEL, dc.solid_round(40), 1000, 500)["stress_MPa"],
      b1["stress_MPa"] / 8)

# A tube of the same OD carries more stress and deflects more than a solid bar.
bt = dc.check_bending(STEEL, dc.tube(20, 16), 1000, 500)
check("tube is more stressed than a solid bar of the same OD",
      bt["stress_MPa"] > b1["stress_MPa"],
      f"{bt['stress_MPa']:.1f} > {b1['stress_MPa']:.1f} MPa")
check("tube deflects more than a solid bar of the same OD",
      bt["deflection_mm"] > b1["deflection_mm"])

raises("unknown beam configuration rejected",
       dc.check_bending, STEEL, r20, 1000, 500, "trampoline")
raises("zero span rejected", dc.check_bending, STEEL, r20, 1000, 0)
raises("negative span rejected", dc.check_bending, STEEL, r20, 1000, -500)

print()
print("=" * 74)
print("E. STRENGTH BASIS AND HONESTY GATES")
print("=" * 74)

cer = dc.check_axial(CERAMIC, r20, 10000)
check("brittle material falls back to ultimate strength",
      cer["basis"] == "ultimate", cer["basis"])
close("brittle FoS uses UTS", cer["fos"], 300 / cer["stress_MPa"])
check("brittle fallback is explained, not silent",
      any("brittle" in w.lower() for w in cer["warnings"]))
check("ceramic reports why fatigue is not checked",
      "fatigue" in cer["not_checked"] and
      "ceramic" in cer["not_checked"]["fatigue"].lower(),
      cer["not_checked"].get("fatigue", "")[:60])

pl = dc.check_axial(FROM_CHROMA, r20, 5000)
check("NOT_FOUND sentinel reads as missing, not as a number",
      pl["not_checked"].get("fatigue") is not None)
check("polymer fatigue gate names the endurance-limit reason",
      "endurance limit" in pl["not_checked"]["fatigue"],
      pl["not_checked"]["fatigue"][:60])
check("a real value alongside a sentinel is still used",
      pl["basis"] == "yield" and pl["fos"] is not None)

metal = dc.check_axial(STEEL, r20, 10000)
check("a material with a fatigue limit has no fatigue gate",
      "fatigue" not in metal["not_checked"])

# Gray cast iron stores yield = 0.0 rather than NULL: its key_warnings row says
# "no useful tensile yield - brittle failure in tension", so the zero is a fact
# about the material. Read as a real strength it gives FoS = 0 under any load,
# condemning a material used for engine blocks and machine bases.
GRAY_IRON = {
    "material_id": "TEST_GRAY_IRON",
    "common_name": "Test Gray Iron",
    "material_class": "cast_iron",
    "yield_strength_MPa": 0.0,
    "ultimate_tensile_strength_MPa": 207.0,
    "elastic_modulus_GPa": 100.0,
    "fatigue_limit_MPa": 90.0,
    "max_service_temp_C": 350.0,
}
gi = dc.check_axial(GRAY_IRON, r20, 10000)
check("a stored yield of 0 is read as 'no yield point', not as 0 MPa",
      gi["basis"] == "ultimate", gi["basis"])
close("zero-yield material takes its FoS from UTS", gi["fos"],
      207 / gi["stress_MPa"])
check("zero-yield material is not condemned at every load",
      gi["verdict"] == "pass", f"{gi['verdict']} at FoS {gi['fos']:.1f}")
check("a 0 MPa yield is not reported back as a strength",
      gi["yield_MPa"] is None and gi["fos_yield"] is None)
check("the brittle basis is explained for zero-yield too",
      any("brittle" in w.lower() for w in gi["warnings"]))

nostrength = dc.check_axial(
    {"material_id": "X", "common_name": "X", "material_class": "ceramic_oxide"},
    r20, 10000)
check("no strength data gives verdict 'unknown', not a number",
      nostrength["verdict"] == "unknown" and nostrength["fos"] is None)

print()
print("=" * 74)
print("F. TEMPERATURE GUARDS")
print("=" * 74)

hot = dc.check_axial(STEEL, r20, 10000, temp_C=500)   # above 400 C max
check("above max service temp warns the FoS is optimistic",
      any("above" in w and "maximum service" in w for w in hot["warnings"]),
      "; ".join(hot["warnings"])[:70])
cold = dc.check_axial(STEEL, r20, 10000, temp_C=-60)  # below -30 C min
check("below min service temp warns about embrittlement",
      any("embrittle" in w.lower() for w in cold["warnings"]))
ok_t = dc.check_axial(STEEL, r20, 10000, temp_C=200)
check("in-range temperature raises no warning", not ok_t["warnings"])
check("no temperature given raises no temperature warning",
      not dc.check_axial(STEEL, r20, 10000)["warnings"])
check("the temperature warning does not silently derate the strength",
      hot["strength_MPa"] == ok_t["strength_MPa"] == 250.0)

print()
print("=" * 74)
print("G. AGAINST THE REAL DATABASE  (all 66 rows)")
print("=" * 74)

import db

conn = db.connect()
materials = db.fetch_materials(conn)
conn.close()
print(f"  loaded {len(materials)} materials")

errors, yielded, ultimate, unknown, no_fatigue, no_deflection = [], [], [], [], [], []
for m in materials:
    try:
        r = dc.check_bending(m, r20, 500, 300, "cantilever_end")
    except Exception as exc:
        errors.append((m.get("material_id"), f"{type(exc).__name__}: {exc}"))
        continue
    {"yield": yielded, "ultimate": ultimate, None: unknown}[r["basis"]].append(m)
    if "fatigue" in r["not_checked"]:
        no_fatigue.append(m)
    if r["deflection_mm"] is None:
        no_deflection.append(m)

check("every material runs without raising", not errors,
      "; ".join(f"{i}: {e}" for i, e in errors[:3]))
check("62 rows check against yield", len(yielded) == 62, f"got {len(yielded)}")
check("4 rows with no yield point fall back to ultimate", len(ultimate) == 4,
      ", ".join(m["material_id"] for m in ultimate))
check("the 4 are the 3 ceramics plus gray cast iron",
      {m["material_id"] for m in ultimate} ==
      {"cer_alumina_96", "cer_sic", "cer_borosilicate_glass", "ci_gray_class30"},
      ", ".join(sorted(m["material_id"] for m in ultimate)))
check("no material is left without any strength basis", not unknown,
      ", ".join(m["material_id"] for m in unknown))
check("19 rows are gated out of the fatigue check", len(no_fatigue) == 19,
      f"got {len(no_fatigue)}")
check("every material has a modulus, so deflection always computes",
      not no_deflection, ", ".join(m["material_id"] for m in no_deflection))

# The fatigue gate must track the material family, not a stray null.
gated_classes = {str(m.get("material_class", "")).split("_")[0] for m in no_fatigue}
check("fatigue gate hits only polymers, composites and ceramics",
      gated_classes <= {"plastic", "composite", "ceramic"},
      ", ".join(sorted(gated_classes)))

# Ranking: strongest first, unknowns last.
ranked = dc.check_many(materials, kind="axial", section=r20, force_N=20000)
foss = [r["fos"] for r in ranked if r["fos"] is not None]
check("check_many ranks by factor of safety, descending",
      foss == sorted(foss, reverse=True))
check("check_many returns every material", len(ranked) == len(materials))
# A FoS of exactly zero means a strength of zero got through as if it were real.
check("no material scores a factor of safety of zero",
      all(r["fos"] > 0 for r in ranked if r["fos"] is not None),
      ", ".join(r["common_name"] for r in ranked if r["fos"] == 0))
print(f"  strongest here: {ranked[0]['common_name']} (FoS {ranked[0]['fos']:.1f})")
print(f"  weakest here:   {ranked[-1]['common_name']} (FoS {ranked[-1]['fos']:.2f})")
raises("check_many rejects an unknown check kind",
       dc.check_many, materials, "torsion", section=r20, force_N=1000)

print()
print("=" * 74)
print(f"RESULT: {len(FAILS)} failure(s)")
for name, detail in FAILS:
    print(f"  - {name}: {detail}")
print("=" * 74)
sys.exit(1 if FAILS else 0)
