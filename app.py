# app.py
# Interactive Streamlit UI for the Material Selection Assistant.
# Renders ranked material cards with color-coded ratings, comparison table,
# pipeline transparency, and chat history.

import streamlit as st
from reason import reason_about_query

# =================================================================
# PAGE CONFIG
# =================================================================
st.set_page_config(
    page_title="Material Selection Assistant",
    page_icon="🔧",
    layout="wide",
)

# =================================================================
# CUSTOM CSS
# =================================================================
st.markdown("""
<style>
/* Calm, low-contrast design — muted neutrals, generous spacing, few accents. */
.material-card {
    background-color: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 1.5rem 1.7rem;
    margin-bottom: 1.1rem;
}
.sec {
    font-size: 1.02rem;
    font-weight: 600;
    color: #c7ccd3;
    letter-spacing: 0.01em;
    margin: 1.4rem 0 0.5rem;
}
.rank-badge {
    display: inline-block;
    padding: 0.12rem 0.55rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.02em;
    margin-right: 0.5rem;
    background-color: rgba(255, 255, 255, 0.07);
    color: #aeb4bd;
    vertical-align: middle;
}
.prop-pill {
    display: inline-block;
    padding: 0.28rem 0.65rem;
    margin: 0.2rem 0.3rem 0.2rem 0;
    background-color: rgba(255, 255, 255, 0.045);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    font-size: 0.8rem;
    color: #c0c5cc;
}
/* Unified subtle note (stress lookups, reasoning) */
.note {
    background-color: rgba(255, 255, 255, 0.025);
    border-left: 3px solid rgba(255, 255, 255, 0.16);
    padding: 0.7rem 1rem;
    border-radius: 4px;
    margin-top: 0.7rem;
    font-size: 0.9rem;
    color: #c0c5cc;
}
.note-accent { border-left-color: #7f9ea6; }
.warn {
    background-color: rgba(205, 160, 95, 0.06);
    border-left: 3px solid rgba(205, 160, 95, 0.55);
    padding: 0.7rem 1rem;
    border-radius: 4px;
    margin-top: 0.7rem;
    font-size: 0.88rem;
    color: #c9b491;
}
/* Family safety banners — visible but not alarming */
.banner {
    padding: 0.6rem 0.9rem;
    border-radius: 6px;
    margin: 0.5rem 0;
    font-size: 0.86rem;
    font-weight: 500;
}
.banner-danger { background: rgba(190, 110, 110, 0.10); border-left: 3px solid #be6e6e; color: #d29a9a; }
.banner-warn   { background: rgba(200, 150, 90, 0.10);  border-left: 3px solid #c8964a; color: #cbab82; }
.muted { color: #8b9098; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# =================================================================
# HELPER FUNCTIONS
# =================================================================
# -----------------------------------------------------------------
# Value coercion — v3 data carries floats (e.g. 5.0) and the string
# "NOT_FOUND" for properties that don't apply to a material's family.
# -----------------------------------------------------------------
def is_na(v):
    """True when a value is missing or the NOT_FOUND sentinel."""
    return v is None or (isinstance(v, str) and v.strip().upper() in ("", "NOT_FOUND"))


def num(v):
    """Float, or None if missing / NOT_FOUND."""
    if is_na(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def rint(v):
    """A 1-5 rating as an int, or None."""
    n = num(v)
    return int(round(n)) if n is not None else None


def fmt(v, suffix="", nd=0):
    """Format a numeric value with an optional unit suffix, or 'N/A'."""
    n = num(v)
    if n is None:
        return "N/A"
    if nd == 0 and float(n).is_integer():
        return f"{int(n)}{suffix}"
    return f"{n:.{nd}f}{suffix}"


def is_family(mat, prefix):
    """True if the material's class starts with prefix (e.g. 'ceramic_')."""
    return str(mat.get("material_class", "")).startswith(prefix)


def color_for_rating(value):
    """Muted color for a 1-5 rating (desaturated, easy on the eyes)."""
    v = rint(value)
    if v is None:
        return "#7c8088"
    v = max(1, min(5, v))
    return ["#b08080", "#b09280", "#b3ad80", "#9bb084", "#86ab8c"][v - 1]


def rating_bar(value, max_value=5):
    """A small muted progress bar for a 1-5 rating, or 'N/A'."""
    v = rint(value)
    if v is None:
        return '<span class="muted">N/A</span>'
    pct = int(round(max(0, min(v, max_value)) / max_value * 100))
    c = color_for_rating(v)
    return (
        '<div style="display:flex;align-items:center;gap:0.5rem;">'
        '<div style="flex:1;height:6px;border-radius:3px;background:rgba(255,255,255,0.06);overflow:hidden;">'
        f'<div style="width:{pct}%;height:100%;background:{c};border-radius:3px;"></div></div>'
        f'<span class="muted" style="min-width:1.8rem;text-align:right;">{v}/5</span></div>'
    )


def rank_badge(rank):
    """A single muted rank chip — no gold/silver/bronze."""
    return f'<span class="rank-badge">#{rank}</span>'


# =================================================================
# SESSION STATE
# =================================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =================================================================
# HEADER
# =================================================================
st.title("Material Selection Assistant")
st.caption(
    "Describe your design conditions — get ranked materials with verified "
    "properties and citations. Press Enter to search."
)


# =================================================================
# SIDEBAR
# =================================================================
with st.sidebar:
    st.caption(
        "Every property value comes from a verified source. "
        "The model only reasons — it cannot invent numbers."
    )

    top_k = st.slider("Recommendations to show", 2, 5, 3)

    st.markdown("---")
    st.markdown("**Try a query**")
    examples = [
        "Lightweight material for marine use",
        "Shaft at 300 MPa stress and 200°C",
        "Low-friction plastic gear that resists fuels",
        "Brittle electrical insulator for 1000°C",
        "Stiff carbon-fibre material for an aerospace panel",
        "Titanium for medical implants",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            # Fill the search box and run on the next rerun.
            st.session_state.query_input = ex
            st.session_state.run_now = True

    st.markdown("---")
    with st.expander("What's in the database"):
        st.caption(
            "66 materials • 24 classes\n\n"
            "**Metals:** Stainless • Carbon • Alloy • Tool steels • "
            "Aluminum • Magnesium • Copper • Nickel • Cast iron • Titanium\n\n"
            "**Non-metals:** Thermoplastics • Thermosets • "
            "Ceramics (oxide/carbide/glass) • Composites (CFRP/GFRP/Kevlar)"
        )

    st.caption(f"Queries this session: **{len(st.session_state.history)}**")
    if st.button("Clear history", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_result = None
        st.rerun()


# =================================================================
# INPUT  (a form so pressing Enter submits — no mouse needed)
# =================================================================
with st.form("query_form", clear_on_submit=False):
    user_query = st.text_input(
        "Your query",
        key="query_input",
        placeholder="e.g., material for a propeller shaft in seawater handling 250 MPa",
        label_visibility="collapsed",
    )
    submit = st.form_submit_button("Find materials", type="primary")


# =================================================================
# RUN PIPELINE
# =================================================================
# Run on Enter / button, or when a sidebar example was clicked.
run = submit or st.session_state.pop("run_now", False)

if run:
    if not user_query.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Searching database and generating recommendations…"):
            result = reason_about_query(user_query, top_k=top_k)

        st.session_state.last_result = result

        # Save to history (only successful retrievals)
        if not result.get("error"):
            st.session_state.history.append({
                "query": user_query,
                "result": result,
            })


# =================================================================
# RENDER RESULT
# =================================================================
def render_result(result):
    # --- Error case ---
    if result.get("error"):
        st.error(f"⚠️ {result['error']}")
        return
    
    # --- Service warning (e.g. Gemini was degraded) ---
    if result.get("warning"):
        st.caption(f"Note: {result['warning']}")

    # --- Overall summary ---
    summary = result.get("summary", "")
    if summary:
        st.markdown('<div class="sec">Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="note note-accent">{summary}</div>', unsafe_allow_html=True)

    # --- Query understanding (collapsible) ---
    with st.expander("How your query was interpreted"):
        understood = result.get("understood", {})
        st.markdown(f"**Reframed as:** *{understood.get('semantic_query', '')}*")
        
        constraints = understood.get('extracted_constraints', {})
        if any(constraints.values()):
            st.markdown("**Detected constraints:**")
            for k, v in constraints.items():
                if v:
                    st.markdown(f"- {k.replace('_', ' ').title()}: `{v}`")
        
        filters = understood.get('filters', {})
        if filters:
            st.markdown(f"**Database filters applied:** `{filters}`")
    
    # --- Comparison table (only if multiple materials) ---
    materials = result.get("materials", [])
    if len(materials) > 1:
        st.markdown('<div class="sec">Comparison</div>', unsafe_allow_html=True)
        cols = st.columns(len(materials))
        for i, mat in enumerate(materials):
            with cols[i]:
                st.markdown(f"**{mat.get('common_name', '')}**")
                st.caption(f"#{i+1} Match")
                # Ceramics/glass have no yield point — fall back to UTS (flexural).
                if is_na(mat.get("yield_strength_MPa")) and not is_na(mat.get("ultimate_tensile_strength_MPa")):
                    st.metric("Flexural (UTS)", fmt(mat.get("ultimate_tensile_strength_MPa"), " MPa"))
                else:
                    st.metric("Yield", fmt(mat.get("yield_strength_MPa"), " MPa"))
                st.metric("Max Temp", fmt(mat.get("max_service_temp_C"), "°C"))
                st.metric("Density", fmt(mat.get("density_kg_m3"), " kg/m³"))

                stress_info = result.get("stress_lookups", {}).get(mat.get("material_id"))
                if stress_info:
                    interp_label = " (interp)" if stress_info.get("interpolated") else ""
                    st.metric(
                        f"Allow. stress @ {stress_info.get('temperature_C')}°C",
                        f"{stress_info.get('stress_MPa')} MPa{interp_label}"
                    )
    
    # --- Material cards ---
    st.markdown('<div class="sec">Recommendations</div>', unsafe_allow_html=True)

    for i, mat in enumerate(materials):
        rank = i + 1
        mat_id = mat.get("material_id")
        
        with st.container():
            st.markdown('<div class="material-card">', unsafe_allow_html=True)
            
            # Header with rank badge
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(
                    f'{rank_badge(rank)} **{mat.get("common_name", "")}**',
                    unsafe_allow_html=True
                )
                st.caption(
                    f"`{mat_id}` • "
                    f"{mat.get('material_class', '').replace('_', ' ').title()} • "
                    f"{mat.get('condition', '').replace('_', ' ')}"
                )
            with col_b:
                relevance = mat.get("relevance_score", 0)
                st.caption(f"Relevance: {relevance:.2f}")

            # --- Family safety banners (visible but calm) ---
            if is_family(mat, "ceramic_"):
                st.markdown(
                    '<div class="banner banner-danger">'
                    'Brittle — design for compression, not tension. Zero ductility; '
                    'fails without warning.</div>',
                    unsafe_allow_html=True,
                )
            if is_family(mat, "composite_"):
                extra = (" Galvanic isolation from aluminium is required."
                         if mat.get("material_class") == "composite_cfrp" else "")
                st.markdown(
                    '<div class="banner banner-warn">'
                    'Directional properties — quoted strength is along the fibre direction; '
                    'transverse strength is far lower. Design the layup for the real load path.'
                    + extra + '</div>',
                    unsafe_allow_html=True,
                )

            # --- Property pills (family-aware: Shore D vs HB, flexural vs yield, etc.) ---
            pills = []
            if is_na(mat.get("yield_strength_MPa")) and not is_na(mat.get("ultimate_tensile_strength_MPa")):
                # Ceramics/glass: no yield point — UTS is the flexural strength.
                pills.append(f"Flexural {fmt(mat.get('ultimate_tensile_strength_MPa'), ' MPa')}")
            else:
                pills.append(f"Yield {fmt(mat.get('yield_strength_MPa'), ' MPa')}")
                if not is_na(mat.get("ultimate_tensile_strength_MPa")):
                    pills.append(f"UTS {fmt(mat.get('ultimate_tensile_strength_MPa'), ' MPa')}")
            if not is_na(mat.get("hardness_HB")):
                pills.append(f"Hardness {fmt(mat.get('hardness_HB'))} HB")
            elif not is_na(mat.get("hardness_shore_d")):
                pills.append(f"Shore D {fmt(mat.get('hardness_shore_d'))}")
            if is_family(mat, "plastic_") and not is_na(mat.get("max_continuous_use_temp_C")):
                pills.append(f"Cont. use {fmt(mat.get('max_continuous_use_temp_C'), '°C')}")
            else:
                pills.append(f"Max temp {fmt(mat.get('max_service_temp_C'), '°C')}")
            pills.append(f"{fmt(mat.get('density_kg_m3'), ' kg/m³')}")
            if not is_na(mat.get("approx_cost_usd_per_kg")):
                pills.append(f"${mat.get('approx_cost_usd_per_kg')}/kg")
            if not is_na(mat.get("machinability_index")):
                pills.append(f"Machinability {fmt(mat.get('machinability_index'))}/100")
            pills_html = "".join(f'<span class="prop-pill">{p}</span>' for p in pills)
            st.markdown(pills_html, unsafe_allow_html=True)

            st.write("")  # spacer
            
            # Corrosion / general ratings grid (5 columns)
            def render_rating_cell(col, label, val):
                with col:
                    st.markdown(
                        f'<div class="muted" style="margin-bottom:0.25rem;">{label}</div>'
                        + rating_bar(val),
                        unsafe_allow_html=True,
                    )

            rating_cols = st.columns(5)
            ratings = [
                ("Seawater", mat.get("corrosion_seawater")),
                ("Acidic", mat.get("corrosion_acidic")),
                ("Alkaline", mat.get("corrosion_alkaline")),
                ("Weldability", mat.get("weldability")),
                ("Fatigue", mat.get("fatigue_rating")),
            ]
            st.markdown('<div class="muted" style="margin-top:0.3rem;">Corrosion &amp; general (1–5)</div>',
                        unsafe_allow_html=True)
            for col, (label, val) in zip(rating_cols, ratings):
                render_rating_cell(col, label, val)

            # Chemical resistance grid (non-metals only — shown when any value exists)
            chem = [
                ("Solvents", mat.get("chemical_resistance_solvents")),
                ("Acids", mat.get("chemical_resistance_acids")),
                ("Alkalis", mat.get("chemical_resistance_alkalis")),
                ("Fuels", mat.get("chemical_resistance_fuels")),
            ]
            if any(not is_na(v) for _, v in chem):
                st.markdown('<div class="muted" style="margin-top:0.6rem;">Chemical resistance (1–5)</div>',
                            unsafe_allow_html=True)
                chem_cols = st.columns(4)
                for col, (label, val) in zip(chem_cols, chem):
                    render_rating_cell(col, label, val)

            # Joining methods + flammability (UL94) chips
            chips = []
            jm = mat.get("joining_method")
            if not is_na(jm):
                for method in str(jm).split("|"):
                    chips.append(f'<span class="prop-pill">{method.replace("_", " ")}</span>')
            flam = mat.get("flammability")
            if not is_na(flam):
                fc = {"V-0": "#86ab8c", "V-1": "#9bb084", "V-2": "#b3ad80", "HB": "#b09280"}.get(str(flam), "#7c8088")
                chips.append(
                    f'<span class="prop-pill" style="border-color:{fc};color:{fc};">'
                    f'UL94 {flam}</span>'
                )
            if not is_na(mat.get("uv_resistance")):
                chips.append(f'<span class="prop-pill">UV {rint(mat.get("uv_resistance"))}/5</span>')
            if chips:
                st.write("")
                st.markdown("".join(chips), unsafe_allow_html=True)

            # Stress info if applicable
            stress_info = result.get("stress_lookups", {}).get(mat_id)
            if stress_info:
                interp_note = " (interpolated)" if stress_info.get("interpolated") else " (exact)"
                st.markdown(
                    f'<div class="note">'
                    f'<b>Allowable stress at {stress_info.get("temperature_C")}°C:</b> '
                    f'{stress_info.get("stress_MPa")} MPa{interp_note}<br>'
                    f'<span class="muted">Source: {stress_info.get("source")}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # LLM reasoning
            reasoning = result.get("reasoning", {}).get(mat_id, "")
            if reasoning:
                st.markdown(
                    f'<div class="note note-accent"><b>Why this fits:</b> {reasoning}</div>',
                    unsafe_allow_html=True
                )

            # Warnings
            warnings = mat.get("key_warnings", "")
            if not is_na(warnings):
                st.markdown(
                    f'<div class="warn"><b>Watch out for:</b> {warnings}</div>',
                    unsafe_allow_html=True
                )
            
            # Expandable details (applications, joining notes, sources)
            with st.expander("More details"):
                if not is_na(mat.get("typical_applications")):
                    st.markdown(f"**Typical applications:** {mat['typical_applications']}")
                if not is_na(mat.get("joining_method")):
                    st.markdown(f"**Joining methods:** {str(mat['joining_method']).replace('|', ', ').replace('_', ' ')}")
                if not is_na(mat.get("weldability_notes")):
                    st.markdown(f"**Joining / welding notes:** {mat['weldability_notes']}")
                if not is_na(mat.get("water_absorption_percent")):
                    st.markdown(f"**Water absorption (ASTM D570):** {fmt(mat.get('water_absorption_percent'), ' %', nd=2)}")
                if not is_na(mat.get("max_continuous_use_temp_C")):
                    st.markdown(f"**Max continuous use temp:** {fmt(mat.get('max_continuous_use_temp_C'), ' °C')}")
                st.markdown(f"**Sources:** `{mat.get('sources', '')}`")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.write("")  # spacer between cards


# --- Show current result ---
if st.session_state.last_result:
    render_result(st.session_state.last_result)


# --- Show history (excluding most recent, already shown above) ---
if st.session_state.history and len(st.session_state.history) > 1:
    st.markdown("---")
    st.markdown('<div class="sec">Previous queries</div>', unsafe_allow_html=True)
    for i, entry in enumerate(reversed(st.session_state.history[:-1])):
        query_num = len(st.session_state.history) - i - 1
        title = entry['query'][:80] + ('...' if len(entry['query']) > 80 else '')
        with st.expander(f"Q{query_num}: {title}"):
            render_result(entry['result'])


# --- Footer ---
st.markdown("---")
st.caption(
    "Recommendations based on a curated database of engineering references. "
    "For critical applications, validate with a qualified engineer."
)