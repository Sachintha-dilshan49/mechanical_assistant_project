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
.material-card {
    background-color: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.rank-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.875rem;
    margin-right: 0.5rem;
}
.rank-1 { background-color: #FFD700; color: #1a1a1a; }
.rank-2 { background-color: #C0C0C0; color: #1a1a1a; }
.rank-3 { background-color: #CD7F32; color: #1a1a1a; }
.rank-other { background-color: rgba(255,255,255,0.1); color: #888; }

.prop-pill {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    margin: 0.2rem 0.3rem 0.2rem 0;
    background-color: rgba(100, 150, 255, 0.15);
    border-radius: 16px;
    font-size: 0.875rem;
    border: 1px solid rgba(100, 150, 255, 0.3);
}
.warning-box {
    background-color: rgba(255, 165, 0, 0.08);
    border-left: 3px solid #ff9800;
    padding: 0.75rem 1rem;
    border-radius: 4px;
    margin-top: 0.75rem;
    font-size: 0.9rem;
}
.reasoning-box {
    background-color: rgba(76, 175, 80, 0.08);
    border-left: 3px solid #4caf50;
    padding: 0.75rem 1rem;
    border-radius: 4px;
    margin-top: 0.75rem;
}
.source-tag {
    font-size: 0.8rem;
    color: #888;
    margin-top: 0.5rem;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)


# =================================================================
# HELPER FUNCTIONS
# =================================================================
def rating_emoji(value, max_value=5):
    """Convert 1-5 rating to colored circles."""
    if value is None:
        return "—"
    filled = "🟢" * value
    empty = "⚪" * (max_value - value)
    return filled + empty


def rating_label(value):
    """Convert 1-5 rating to text label."""
    labels = ["—", "Poor", "Fair", "Moderate", "Good", "Excellent"]
    return labels[value] if value else "—"


def rank_badge(rank):
    """Return HTML for the rank badge (gold/silver/bronze/etc)."""
    if rank == 1:
        return '<span class="rank-badge rank-1">🥇 #1 Match</span>'
    elif rank == 2:
        return '<span class="rank-badge rank-2">🥈 #2 Match</span>'
    elif rank == 3:
        return '<span class="rank-badge rank-3">🥉 #3 Match</span>'
    else:
        return f'<span class="rank-badge rank-other">#{rank}</span>'


def color_for_rating(value):
    """Return color based on 1-5 rating."""
    if value is None:
        return "#888"
    return ["#ff5252", "#ff9800", "#ffc107", "#8bc34a", "#4caf50"][value - 1]


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
st.title("🔧 Mechanical Material Selection Assistant")
st.markdown(
    "Describe your design conditions. Get ranked materials with verified data "
    "from Shigley, ASME, Cardarelli, Outokumpu, ESAB, and Machinery's Handbook."
)


# =================================================================
# SIDEBAR
# =================================================================
with st.sidebar:
    st.header("About")
    st.caption(
        "Every property value comes from a verified source. "
        "The LLM only reasons — it cannot invent numbers."
    )
    
    st.markdown("---")
    st.subheader("Try a query")
    examples = [
        "Lightweight material for marine use",
        "Shaft at 300 MPa stress and 200°C",
        "Bearing material for a slow pump",
        "Engine block material",
        "Cutting die that won't dull",
        "Titanium for medical implants",
        "Cheapest free-machining metal",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.example_query = ex
    
    st.markdown("---")
    st.subheader("Database")
    st.caption(
        "30 materials • 14 classes\n\n"
        "Stainless • Carbon • Alloy • Tool steels • "
        "Aluminum • Copper • Cast iron • Titanium"
    )
    
    st.markdown("---")
    st.caption(f"📜 Queries this session: **{len(st.session_state.history)}**")
    if st.button("Clear history", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_result = None
        st.rerun()


# =================================================================
# INPUT
# =================================================================
default_query = st.session_state.pop("example_query", "")

with st.container():
    user_query = st.text_area(
        "Your query",
        value=default_query,
        placeholder="e.g., I need a material for a propeller shaft in seawater that can handle 250 MPa",
        height=80,
        label_visibility="collapsed",
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        top_k = st.slider("Number of recommendations", 2, 5, 3)
    with col2:
        st.write("")  # spacer
        submit = st.button("🔍 Find Materials", type="primary", use_container_width=True)


# =================================================================
# RUN PIPELINE
# =================================================================
if submit:
    if not user_query.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Analyzing query, searching database, generating recommendations..."):
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
        st.warning(f"⚠️ {result['warning']}")
    
    # --- Overall summary ---
    st.markdown("### 📋 Recommendation Summary")
    st.info(result.get("summary", ""))
    
    # --- Query understanding (collapsible) ---
    with st.expander("🧠 How the system interpreted your query"):
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
        st.markdown("### 📊 At-a-Glance Comparison")
        cols = st.columns(len(materials))
        for i, mat in enumerate(materials):
            with cols[i]:
                st.markdown(f"**{mat.get('common_name', '')}**")
                st.caption(f"#{i+1} Match")
                st.metric("Yield", f"{mat.get('yield_strength_MPa', 'N/A')} MPa")
                st.metric("Max Temp", f"{mat.get('max_service_temp_C', 'N/A')}°C")
                st.metric("Density", f"{mat.get('density_kg_m3', 'N/A')} kg/m³")
                
                stress_info = result.get("stress_lookups", {}).get(mat.get("material_id"))
                if stress_info:
                    interp_label = " (interp)" if stress_info.get("interpolated") else ""
                    st.metric(
                        f"Allow. stress @ {stress_info.get('temperature_C')}°C",
                        f"{stress_info.get('stress_MPa')} MPa{interp_label}"
                    )
    
    # --- Material cards ---
    st.markdown("### 🔬 Detailed Recommendations")
    
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
            
            # Property pills
            pills = [
                f"💪 Yield: {mat.get('yield_strength_MPa')} MPa",
                f"💯 UTS: {mat.get('ultimate_tensile_strength_MPa')} MPa",
                f"🌡️ Max: {mat.get('max_service_temp_C')}°C",
                f"⚖️ {mat.get('density_kg_m3')} kg/m³",
                f"💲 Cost: {mat.get('approx_cost_usd_per_kg', '')} USD/kg",
                f"🔧 Mach: {mat.get('machinability_index')}/100",
            ]
            pills_html = "".join(f'<span class="prop-pill">{p}</span>' for p in pills)
            st.markdown(pills_html, unsafe_allow_html=True)
            
            st.write("")  # spacer
            
            # Ratings grid (5 columns)
            rating_cols = st.columns(5)
            ratings = [
                ("Seawater corrosion", mat.get("corrosion_seawater")),
                ("Acidic corrosion", mat.get("corrosion_acidic")),
                ("Alkaline corrosion", mat.get("corrosion_alkaline")),
                ("Weldability", mat.get("weldability")),
                ("Fatigue", mat.get("fatigue_rating")),
            ]
            for col, (label, val) in zip(rating_cols, ratings):
                with col:
                    st.markdown(f"**{label}**")
                    if val is not None:
                        st.markdown(
                            f'<div style="color:{color_for_rating(val)};font-weight:600;">'
                            f'{rating_emoji(val)}<br>{rating_label(val)} ({val}/5)'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.caption("—")
            
            # Stress info if applicable
            stress_info = result.get("stress_lookups", {}).get(mat_id)
            if stress_info:
                interp_note = " (interpolated)" if stress_info.get("interpolated") else " (exact)"
                st.markdown(
                    f'<div class="reasoning-box">'
                    f'<b>⚡ Allowable stress at {stress_info.get("temperature_C")}°C:</b> '
                    f'{stress_info.get("stress_MPa")} MPa{interp_note}<br>'
                    f'<small>Source: {stress_info.get("source")}</small>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # LLM reasoning
            reasoning = result.get("reasoning", {}).get(mat_id, "")
            if reasoning:
                st.markdown(
                    f'<div class="reasoning-box">'
                    f'<b>✅ Why this fits:</b> {reasoning}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Warnings
            warnings = mat.get("key_warnings", "")
            if warnings:
                st.markdown(
                    f'<div class="warning-box">'
                    f'<b>⚠️ Watch out for:</b> {warnings}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Expandable details (applications, welding notes, sources)
            with st.expander("More details"):
                if mat.get("typical_applications"):
                    st.markdown(f"**Typical applications:** {mat['typical_applications']}")
                if mat.get("weldability_notes"):
                    st.markdown(f"**Welding notes:** {mat['weldability_notes']}")
                st.markdown(f"**Sources:** `{mat.get('sources', '')}`")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.write("")  # spacer between cards


# --- Show current result ---
if st.session_state.last_result:
    render_result(st.session_state.last_result)


# --- Show history (excluding most recent, already shown above) ---
if st.session_state.history and len(st.session_state.history) > 1:
    st.markdown("---")
    st.markdown("### 📜 Previous queries this session")
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