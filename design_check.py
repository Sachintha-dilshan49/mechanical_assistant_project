"""
design_check.py
Deterministic strength checks — "will this part hold?" for a candidate material.

This module does arithmetic, not judgement. No LLM call is made here and none
may ever be added: a model that miscomputes a stress produces a confident wrong
factor of safety, which is worse than no answer at all. Same rule as unit
conversion in the ingester — Python does the maths, the model only talks about
the result.

Every strength number comes from SQLite via the material dict handed in. This
file supplies only textbook mechanics (Shigley, "Mechanical Engineering
Design"): section properties, direct stress, bending stress and beam deflection.

UNITS — the single most dangerous thing in this file, so it is fixed and total:

    lengths and dimensions   mm
    forces                   N
    distributed loads        N/mm
    moments                  N*mm   (computed internally, never an input)
    stress and strength      MPa    ( = N/mm^2, which is why mm and N are used )
    modulus                  MPa    (converted from the GPa stored in SQLite)
    deflection               mm

Because 1 N/mm^2 is exactly 1 MPa, working in N and mm means a stress falls out
in MPa with no conversion factor anywhere. Do not introduce metres.

Scope: axial and bending. Torsion, column buckling and the fatigue overlay are
deliberately absent rather than approximated — see the family gates in
`unsupported_checks()` for what a given material can and cannot be checked for.
"""

import math

# The 1-5 rating sentinel used across the app for "does not apply".
NOT_FOUND = "NOT_FOUND"

# Default target factor of safety. 1.5 is the usual teaching starting point for
# ductile materials under static, well-known loads; real targets are set by the
# application and its code, so this is a parameter, not a truth.
DEFAULT_TARGET_FOS = 1.5


# =================================================================
# Value access — the same NOT_FOUND handling the rest of the app uses
# =================================================================
def _num(m, key):
    """A numeric field as a float, or None when missing / NOT_FOUND.

    Material dicts reach here from two places: straight out of SQLite (real
    NULLs) and back out of ChromaDB (the NOT_FOUND sentinel, because Chroma
    metadata cannot hold nulls). Both must read as "no value".
    """
    v = m.get(key)
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.upper() == NOT_FOUND:
        return None
    try:
        return float(s)
    except ValueError:
        return None


# =================================================================
# SECTION PROPERTIES
# A section is the shape of the cross-section being loaded. These are exact
# closed-form values, not approximations.
#
#   A   cross-sectional area                 mm^2
#   I   second moment of area about bending  mm^4
#   c   distance from neutral axis to the extreme fibre  mm
#   J   polar second moment of area          mm^4, or None where torsion of
#       that shape is not a closed form (rectangles need a series solution,
#       which belongs with the torsion check, not here)
# =================================================================
def solid_round(d_mm):
    """Solid circular bar of diameter d."""
    d = _positive(d_mm, "diameter")
    return {
        "name": f"solid round d={_g(d)} mm",
        "A_mm2": math.pi * d ** 2 / 4,
        "I_mm4": math.pi * d ** 4 / 64,
        "c_mm": d / 2,
        "J_mm4": math.pi * d ** 4 / 32,
    }


def tube(od_mm, id_mm):
    """Hollow circular tube of outer diameter od and inner diameter id."""
    od = _positive(od_mm, "outer diameter")
    idia = _positive(id_mm, "inner diameter", allow_zero=True)
    if idia >= od:
        raise ValueError(
            f"inner diameter ({_g(idia)} mm) must be smaller than outer "
            f"diameter ({_g(od)} mm)"
        )
    return {
        "name": f"tube {_g(od)}x{_g(idia)} mm",
        "A_mm2": math.pi * (od ** 2 - idia ** 2) / 4,
        "I_mm4": math.pi * (od ** 4 - idia ** 4) / 64,
        "c_mm": od / 2,
        "J_mm4": math.pi * (od ** 4 - idia ** 4) / 32,
    }


def rectangle(b_mm, h_mm):
    """Solid rectangle, width b, height h. Bending is about the strong axis,
    i.e. h is measured in the direction the load is applied."""
    b = _positive(b_mm, "width")
    h = _positive(h_mm, "height")
    return {
        "name": f"rectangle {_g(b)}x{_g(h)} mm",
        "A_mm2": b * h,
        "I_mm4": b * h ** 3 / 12,
        "c_mm": h / 2,
        # Torsion of a rectangle is not I-based; it needs the beta-series
        # solution. Left None so the torsion check must handle it explicitly
        # rather than silently using a wrong number.
        "J_mm4": None,
    }


def _positive(v, label, allow_zero=False):
    """Validate a dimension. Bad geometry must fail loudly at entry — a
    negative or zero dimension otherwise reaches the division and produces
    either a crash deep in the maths or a nonsensical stress."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number, got {v!r}")
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"{label} must be a finite number, got {v!r}")
    if f < 0 or (f == 0 and not allow_zero):
        raise ValueError(f"{label} must be greater than zero, got {_g(f)}")
    return f


def _g(v):
    """Compact number for a label: 20 not 20.0, 12.5 stays 12.5."""
    f = float(v)
    return str(int(f)) if f.is_integer() else f"{f:g}"


# =================================================================
# BEAM CONFIGURATIONS
# Maximum bending moment and maximum deflection for the standard cases.
#
#   moment(P, L)      -> N*mm      P is N for point loads, N/mm for a UDL
#   deflect(P, L, EI) -> mm
#
# Keeping the pair together per configuration is deliberate: the moment and the
# deflection for a case must come from the same beam, and splitting them across
# two lookup tables is how they drift apart.
# =================================================================
BEAMS = {
    "cantilever_end": {
        "label": "cantilever, point load at the free end",
        "load_unit": "N",
        "moment": lambda P, L: P * L,
        "deflect": lambda P, L, EI: P * L ** 3 / (3 * EI),
    },
    "cantilever_udl": {
        "label": "cantilever, uniformly distributed load",
        "load_unit": "N/mm",
        "moment": lambda w, L: w * L ** 2 / 2,
        "deflect": lambda w, L, EI: w * L ** 4 / (8 * EI),
    },
    "simple_center": {
        "label": "simply supported, point load at mid-span",
        "load_unit": "N",
        "moment": lambda P, L: P * L / 4,
        "deflect": lambda P, L, EI: P * L ** 3 / (48 * EI),
    },
    "simple_udl": {
        "label": "simply supported, uniformly distributed load",
        "load_unit": "N/mm",
        "moment": lambda w, L: w * L ** 2 / 8,
        "deflect": lambda w, L, EI: 5 * w * L ** 4 / (384 * EI),
    },
}


# =================================================================
# STRENGTH BASIS
# Which strength a factor of safety is measured against, and why.
# =================================================================
def strength_basis(material):
    """Pick the limiting strength for this material.

    Ductile materials are checked against yield: past it the part is
    permanently deformed, which is failure for a machine element even though it
    has not broken. Brittle materials have no yield point, so they are checked
    against ultimate strength and the caller is told the basis changed.

    "No yield point" reaches this function in two encodings, and both must be
    caught. The three ceramics store NULL. Gray cast iron stores 0.0 — its own
    key_warnings row says "no useful tensile yield, brittle failure in tension",
    so the zero is a deliberate statement about the material, not missing data.
    Treating that 0.0 as a real strength yields FoS = 0 at every load, i.e. a
    part that fails while carrying nothing, for a material used in engine
    blocks and machine bases.

    Returns (strength_MPa, basis, note) with strength None if neither is known.
    """
    sy = _num(material, "yield_strength_MPa")
    su = _num(material, "ultimate_tensile_strength_MPa")

    if sy is not None and sy > 0:
        return sy, "yield", None
    if su is not None:
        return su, "ultimate", (
            "No yield strength: this material is brittle and has no yield "
            "point, so the factor of safety is measured against ultimate "
            "strength. A brittle part gives no warning before it breaks — "
            "design to a higher factor of safety than you would for a metal."
        )
    return None, None, "No yield or ultimate strength recorded for this material."


def unsupported_checks(material):
    """Checks that cannot be run for this material, with the reason.

    Stated rather than silently omitted: a missing fatigue row is a fact about
    the material, not a hole in the database, and a student needs to know which
    it is.
    """
    out = {}
    if _num(material, "fatigue_limit_MPa") is None:
        cls = str(material.get("material_class", ""))
        if cls.startswith("plastic"):
            reason = ("polymers have no true endurance limit — their fatigue "
                      "strength keeps falling with cycle count, so a single "
                      "limit value would be misleading")
        elif cls.startswith("composite"):
            reason = ("composites fail in fatigue by progressive delamination "
                      "rather than at a single stress limit")
        elif cls.startswith("ceramic"):
            reason = "brittle ceramics are not characterised by a fatigue limit"
        else:
            reason = "no fatigue limit recorded for this material"
        out["fatigue"] = reason
    return out


# =================================================================
# THE CHECKS
# =================================================================
def check_axial(material, section, force_N, temp_C=None,
                target_fos=DEFAULT_TARGET_FOS):
    """Direct tension or compression: sigma = F / A.

    Compression is checked here against strength only. A slender member in
    compression fails by buckling at a load far below its crush strength, and
    that check does not exist yet — hence the explicit warning below rather
    than a quietly optimistic answer.
    """
    F = abs(_finite(force_N, "force"))
    A = section["A_mm2"]
    sigma = F / A

    result = _assess(material, sigma, temp_C, target_fos)
    result["load"] = {
        "kind": "axial",
        "label": "axial load",
        "force_N": F,
    }
    result["section"] = section
    result["deflection_mm"] = None
    result["assumptions"] = [
        "Static load, applied along the axis of the member.",
        "Stress is uniform across the section — no stress concentration "
        "factor (Kt) is applied, so holes, notches, fillets and threads are "
        "not accounted for.",
    ]
    if float(force_N) < 0:
        result["warnings"].append(
            "Compressive load: this checks crushing strength only. A slender "
            "member buckles well below this load — a buckling check is not "
            "included yet."
        )
    return result


def check_bending(material, section, load, span_mm, beam="cantilever_end",
                  temp_C=None, target_fos=DEFAULT_TARGET_FOS):
    """Bending: sigma = M*c / I, with the maximum deflection for the same beam.

    `load` is N for the point-load configurations and N/mm for the distributed
    ones; `BEAMS[beam]["load_unit"]` says which, and it is echoed back in the
    result so a UI can label the input correctly.
    """
    if beam not in BEAMS:
        raise ValueError(
            f"unknown beam configuration {beam!r}; expected one of "
            + ", ".join(sorted(BEAMS))
        )
    cfg = BEAMS[beam]
    P = abs(_finite(load, "load"))
    L = _positive(span_mm, "span")

    M = cfg["moment"](P, L)                       # N*mm
    sigma = M * section["c_mm"] / section["I_mm4"]

    result = _assess(material, sigma, temp_C, target_fos)
    result["load"] = {
        "kind": "bending",
        "label": cfg["label"],
        "beam": beam,
        "magnitude": P,
        "unit": cfg["load_unit"],
        "span_mm": L,
        "moment_Nmm": M,
    }
    result["section"] = section

    # Deflection needs the modulus; every one of the 66 rows has it, but a
    # material dict assembled elsewhere might not, so it stays optional.
    E_GPa = _num(material, "elastic_modulus_GPa")
    if E_GPa is not None and E_GPa > 0:
        E = E_GPa * 1000.0                        # GPa -> MPa (N/mm^2)
        result["deflection_mm"] = cfg["deflect"](P, L, E * section["I_mm4"])
        result["modulus_MPa"] = E
    else:
        result["deflection_mm"] = None
        result["warnings"].append(
            "No elastic modulus recorded, so deflection cannot be computed."
        )

    result["assumptions"] = [
        f"Beam: {cfg['label']}.",
        "Static load. Linear-elastic, small deflection (Euler-Bernoulli) "
        "theory — valid while the part stays below yield.",
        "Bending about the strong axis, self-weight ignored.",
        "No stress concentration factor (Kt) is applied, so holes, notches "
        "and fillets are not accounted for.",
    ]
    return result


def _assess(material, sigma_MPa, temp_C, target_fos):
    """Common half of every check: compare a computed stress to the material's
    strength and produce the verdict, warnings and provenance."""
    strength, basis, basis_note = strength_basis(material)
    su = _num(material, "ultimate_tensile_strength_MPa")

    # A stored yield of 0 means "no usable yield point" (see strength_basis),
    # so it is reported as absent here rather than as a 0 MPa strength.
    sy = _num(material, "yield_strength_MPa")
    if sy is not None and sy <= 0:
        sy = None

    fos = (strength / sigma_MPa) if (strength is not None and sigma_MPa > 0) else None

    if fos is None:
        verdict = "unknown"
    elif fos < 1.0:
        verdict = "fail"
    elif fos < target_fos:
        verdict = "marginal"
    else:
        verdict = "pass"

    warnings = []
    if basis_note:
        warnings.append(basis_note)

    # Service temperature is stored for all 66 rows, so this guard always has
    # data to work with. It is a warning, not a derating: the ASME allowable
    # stress tables that would let us actually derate cover only 7 materials.
    if temp_C is not None:
        t_max = _num(material, "max_service_temp_C")
        t_min = _num(material, "min_service_temp_C")
        if t_max is not None and temp_C > t_max:
            warnings.append(
                f"{_g(temp_C)} C is above this material's maximum service "
                f"temperature of {_g(t_max)} C. Room-temperature strength does "
                f"not apply and this factor of safety is optimistic."
            )
        if t_min is not None and temp_C < t_min:
            warnings.append(
                f"{_g(temp_C)} C is below this material's minimum service "
                f"temperature of {_g(t_min)} C. Check for low-temperature "
                f"embrittlement."
            )

    return {
        "material_id": material.get("material_id"),
        "common_name": material.get("common_name"),
        "material_class": material.get("material_class"),
        "stress_MPa": sigma_MPa,
        "strength_MPa": strength,
        "basis": basis,
        "yield_MPa": sy,
        "ultimate_MPa": su,
        "fos": fos,
        "fos_yield": (sy / sigma_MPa) if (sy and sigma_MPa > 0) else None,
        "fos_ultimate": (su / sigma_MPa) if (su and sigma_MPa > 0) else None,
        "target_fos": target_fos,
        "verdict": verdict,
        "temperature_C": temp_C,
        "warnings": warnings,
        "not_checked": unsupported_checks(material),
    }


def _finite(v, label):
    """A load value that is a real number. Zero is allowed — an unloaded part
    is a legitimate thing to ask about — but NaN and infinity are not."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number, got {v!r}")
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"{label} must be a finite number, got {v!r}")
    return f


# =================================================================
# Running a check across the candidates the pipeline returned
# =================================================================
def check_many(materials, kind="axial", **kwargs):
    """Run the same load case over a list of materials, best first.

    Ordering is by factor of safety descending, with unknowns last — that is a
    different ranking from the one the reasoning step produces, and it is meant
    to be: this one answers "which of these is strongest here", not "which of
    these suits the job".
    """
    fn = {"axial": check_axial, "bending": check_bending}.get(kind)
    if fn is None:
        raise ValueError(f"unknown check kind {kind!r}; expected 'axial' or 'bending'")
    results = [fn(m, **kwargs) for m in materials]
    return sorted(results, key=lambda r: (r["fos"] is None, -(r["fos"] or 0)))
