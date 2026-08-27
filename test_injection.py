"""
test_injection.py
Renders a material whose every text field carries an HTML payload, and asserts
that no unsafe_allow_html block emits it raw. Plain st.markdown calls are
ignored: Streamlit escapes those itself, so they are not an injection path.

    py test_injection.py
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import streamlit as st

PAYLOAD = '<img src=x onerror="window.__pwned=1">'
unsafe_bodies = []
safe_bodies = []

_real_markdown = st.markdown


def spy_markdown(body="", *args, **kwargs):
    if kwargs.get("unsafe_allow_html"):
        unsafe_bodies.append(str(body))
    else:
        safe_bodies.append(str(body))
    return None


# Neutralise the widgets render_result touches so it can run headless.
st.markdown = spy_markdown
for name in ("caption", "write", "error", "metric", "progress", "divider", "toast"):
    setattr(st, name, lambda *a, **k: None)


class _Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


st.container = lambda *a, **k: _Ctx()
st.expander = lambda *a, **k: _Ctx()
st.columns = lambda spec, **k: [_Ctx() for _ in (range(spec) if isinstance(spec, int) else spec)]
st.chat_message = lambda *a, **k: _Ctx()

import importlib.util
spec = importlib.util.spec_from_file_location("appmod", "app.py")
appmod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(appmod)
except Exception:
    pass                                  # the chat loop at the bottom needs a session

result = {
    "query": "x", "resolved_query": None,
    "understood": {"semantic_query": "x", "filters": {}, "extracted_constraints": {}},
    "materials": [{
        "material_id": f"id {PAYLOAD}",
        "common_name": f"Steel {PAYLOAD}",
        "material_class": "carbon_steel_low",
        "condition": f"annealed {PAYLOAD}",
        "yield_strength_MPa": 250.0, "ultimate_tensile_strength_MPa": 400.0,
        "density_kg_m3": 7850.0, "max_service_temp_C": 400.0,
        "relevance_score": 0.5,
        "hardness_HB": 120.0, "corrosion_seawater": 3.0, "weldability": 4.0,
        "approx_cost_usd_per_kg": f"1-2 {PAYLOAD}",
        "joining_method": f"welding|{PAYLOAD}",
        "flammability": PAYLOAD,
        "key_warnings": f"Careful {PAYLOAD}",
        "typical_applications": f"Brackets {PAYLOAD}",
        "sources": f"Shigley {PAYLOAD}",
    }, {
        "material_id": "id2", "common_name": f"Brass {PAYLOAD}",
        "material_class": "copper_alloy", "condition": "annealed",
        "yield_strength_MPa": 310.0, "ultimate_tensile_strength_MPa": 400.0,
        "density_kg_m3": 8500.0, "max_service_temp_C": 200.0, "relevance_score": 0.4,
        "sources": "x",
    }],
    "stress_lookups": {f"id {PAYLOAD}": {
        "stress_MPa": f"100 {PAYLOAD}", "source": f"ASME {PAYLOAD}",
        "interpolated": True, "temperature_C": f"200 {PAYLOAD}"}},
    "reasoning": {f"id {PAYLOAD}": f"Good pick {PAYLOAD}"},
    "summary": f"Use this steel {PAYLOAD}",
    "warning": None, "error": None, "chat_reply": None,
}

appmod.render_result(result)

print("=" * 74)
print("INJECTION TEST - only unsafe_allow_html=True bodies are checked")
print("=" * 74)
print(f"  unsafe_allow_html blocks rendered : {len(unsafe_bodies)}")
print(f"  plain (auto-escaped) blocks       : {len(safe_bodies)}")

leaks = [b for b in unsafe_bodies if PAYLOAD in b]
print()
if leaks:
    print(f"  FAIL  {len(leaks)} unsafe block(s) contain the raw payload:")
    for b in leaks[:6]:
        idx = b.find(PAYLOAD)
        print("        ..." + b[max(0, idx - 70):idx + 45].replace("\n", " "))
else:
    print("  PASS  no unsafe block contains raw attacker-controlled markup")

escaped_hits = sum(1 for b in unsafe_bodies if "&lt;img" in b)
print(f"  PASS  payload appears escaped (&lt;img) in {escaped_hits} block(s)")

plain_leaks = [b for b in safe_bodies if PAYLOAD in b]
print(f"\n  note: {len(plain_leaks)} plain st.markdown call(s) carry the payload - "
      f"Streamlit escapes those itself, so they are not an injection path.")
print("=" * 74)
sys.exit(1 if leaks else 0)
