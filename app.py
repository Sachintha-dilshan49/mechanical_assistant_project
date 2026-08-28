# app.py
# Interactive Streamlit UI for the Material Selection Assistant.
# Renders ranked material cards with color-coded ratings, comparison table,
# pipeline transparency, and chat history.

import html
import json
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
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
/* =================================================================
   DESIGN TOKENS
   One warm, low-contrast palette. Every colour below is referenced by
   name, so the whole UI can be retuned from this block alone. Values
   mirror .streamlit/config.toml - keep them in sync.
   ================================================================= */
:root {
    --bg:            #FAF9F5;
    --surface:       #FFFFFF;
    --surface-sunk:  #F2F0EA;
    --border:        #E4E1D9;
    --border-strong: #D3CFC4;
    --text:          #262624;
    --text-muted:    #6E6B63;
    --text-faint:    #938F85;
    --accent:        #C96442;
    --accent-wash:   rgba(201, 100, 66, 0.07);
    --radius:        12px;
    --radius-sm:     7px;
}

/* Base typography: generous line height, comfortable measure. */
html, body, [class*="css"] {
    font-feature-settings: "kern" 1, "liga" 1;
    -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--bg); }
.block-container { padding-top: 2.2rem; max-width: 52rem; }

/* Section label - small, tracked, quiet. Replaces the old bold headings. */
.sec {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--text-faint);
    margin: 2rem 0 0.7rem;
}

/* -----------------------------------------------------------------
   MATERIAL CARD
   Flat surface, hairline border, no drop shadow until hover. The old
   card used translucent white on dark, which only worked on one theme.
   ----------------------------------------------------------------- */
.material-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.35rem 1.5rem 1.15rem;
    margin-bottom: 0.9rem;
    transition: border-color 140ms ease, box-shadow 140ms ease;
}
.material-card:hover {
    border-color: var(--border-strong);
    box-shadow: 0 1px 3px rgba(38, 38, 36, 0.05);
}
.mat-name {
    font-size: 1.06rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.01em;
}
.mat-meta {
    font-size: 0.78rem;
    color: var(--text-faint);
    margin-top: 0.15rem;
}

/* Rank chip - the #1 pick gets the accent, the rest stay quiet. */
.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 1.3rem;
    height: 1.3rem;
    padding: 0 0.35rem;
    border-radius: var(--radius-sm);
    font-size: 0.72rem;
    font-weight: 600;
    background: var(--surface-sunk);
    color: var(--text-muted);
    border: 1px solid var(--border);
    margin-right: 0.55rem;
}
.rank-badge.top {
    background: var(--accent-wash);
    color: var(--accent);
    border-color: rgba(201, 100, 66, 0.25);
}

/* Property chips */
.prop-pill {
    display: inline-block;
    padding: 0.24rem 0.55rem;
    margin: 0.22rem 0.3rem 0.22rem 0;
    background: var(--surface-sunk);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-size: 0.78rem;
    color: var(--text-muted);
    white-space: nowrap;
}
.prop-pill b { color: var(--text); font-weight: 600; }

/* Callouts: reasoning, stress data, warnings, safety banners. */
.note {
    background: var(--surface-sunk);
    border-left: 2px solid var(--border-strong);
    padding: 0.75rem 0.95rem;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    margin-top: 0.75rem;
    font-size: 0.88rem;
    line-height: 1.6;
    color: var(--text-muted);
}
.note-accent {
    background: var(--accent-wash);
    border-left-color: var(--accent);
    color: var(--text);
}
.warn {
    background: rgba(193, 138, 62, 0.08);
    border-left: 2px solid #C18A3E;
    padding: 0.75rem 0.95rem;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    margin-top: 0.7rem;
    font-size: 0.86rem;
    line-height: 1.6;
    color: #7A5A24;
}
.banner {
    padding: 0.6rem 0.85rem;
    border-radius: var(--radius-sm);
    margin: 0.7rem 0 0.2rem;
    font-size: 0.84rem;
    line-height: 1.5;
}
.banner-danger { background: rgba(184, 92, 76, 0.08);  border-left: 2px solid #B85C4C; color: #8A3F33; }
.banner-warn   { background: rgba(193, 138, 62, 0.08); border-left: 2px solid #C18A3E; color: #7A5A24; }
.muted { color: var(--text-faint); font-size: 0.75rem; }

/* Data-confidence tag. Says whether a card's numbers came from a primary
   source or were estimated - the free-text `sources` field can't be read at a
   glance, and a student has no other way to tell the two apart. Estimated is
   the loudest of the three on purpose. */
.conf-tag {
    display: inline-block;
    padding: 0.16rem 0.5rem;
    margin: 0.5rem 0 0.1rem;
    border-radius: var(--radius-sm);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    border: 1px solid;
}
.conf-verified { background: rgba(94, 140, 99, 0.10);  border-color: rgba(94, 140, 99, 0.35);  color: #4A7350; }
.conf-cross    { background: rgba(110, 107, 99, 0.08); border-color: var(--border-strong);     color: var(--text-muted); }
.conf-est      { background: rgba(193, 138, 62, 0.10); border-color: rgba(193, 138, 62, 0.40); color: #7A5A24; }

/* -----------------------------------------------------------------
   CHAT
   The user's turn is a quiet outlined block, not a coloured bubble.
   The assistant's turn has no container at all, so the content itself
   carries the page - the same reason chat UIs read as calm.
   ----------------------------------------------------------------- */
.wa-row { display: flex; margin: 1.4rem 0 0.2rem; }
.wa-row.user { justify-content: flex-end; }
.wa-bubble-user {
    background: var(--surface-sunk);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 0.6rem 0.9rem;
    border-radius: var(--radius);
    max-width: 80%;
    font-size: 0.94rem;
    line-height: 1.55;
    overflow-wrap: anywhere;
}

/* Streamlit chrome: soften borders, match the palette. */
[data-testid="stChatMessage"] {
    background: transparent;
    padding: 0.2rem 0 0;
}
[data-testid="stSidebar"] {
    background: var(--surface-sunk);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stButton button {
    text-align: left;
    justify-content: flex-start;
    font-size: 0.85rem;
    font-weight: 450;
    border-radius: var(--radius-sm);
}
[data-testid="stMetricValue"] { font-size: 1.15rem; font-weight: 600; }
[data-testid="stMetricLabel"] { font-size: 0.74rem; color: var(--text-faint); }
[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--surface);
}
[data-testid="stChatInput"] {
    border-radius: var(--radius);
    border-color: var(--border-strong);
}
/* Tertiary buttons are the per-message actions - keep them near-invisible
   until hovered so they do not compete with the answer. */
button[kind="tertiary"] {
    color: var(--text-faint) !important;
    font-size: 0.76rem !important;
    padding: 0.1rem 0.4rem !important;
}
button[kind="tertiary"]:hover { color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# =================================================================
# HELPER FUNCTIONS
# =================================================================
# -----------------------------------------------------------------
# Value coercion — v3 data carries floats (e.g. 5.0) and the string
# "NOT_FOUND" for properties that don't apply to a material's family.
# -----------------------------------------------------------------
def esc(v):
    """HTML-escape a value bound for an unsafe_allow_html block.

    Everything rendered through those blocks comes from either the model or the
    database, and neither is trusted markup: a summary containing a tag was
    being injected straight into the page. Escaping costs literal asterisks if a
    model emits markdown here, which is the right trade for not rendering
    attacker-controlled HTML.
    """
    return html.escape(str(v)) if v is not None else ""


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
    """Colour for a 1-5 rating, tuned for the light background.

    Deliberately desaturated: five saturated traffic-light bars per card
    turns the page into noise when the numbers are the point.
    """
    v = rint(value)
    if v is None:
        return "#B9B5AB"
    v = max(1, min(5, v))
    return ["#BC6B57", "#C08A4E", "#AFA057", "#7F9A63", "#5E8C63"][v - 1]


def rating_bar(value, max_value=5):
    """A slim bar for a 1-5 rating, or 'N/A'."""
    v = rint(value)
    if v is None:
        return '<span class="muted">N/A</span>'
    pct = int(round(max(0, min(v, max_value)) / max_value * 100))
    c = color_for_rating(v)
    return (
        '<div style="display:flex;align-items:center;gap:0.5rem;">'
        '<div style="flex:1;height:5px;border-radius:3px;'
        'background:var(--surface-sunk);border:1px solid var(--border);overflow:hidden;">'
        f'<div style="width:{pct}%;height:100%;background:{c};"></div></div>'
        f'<span class="muted" style="min-width:1.6rem;text-align:right;">{v}/5</span></div>'
    )


# How much to trust a card's numbers. Anything unrecognised - a blank field, or
# a row indexed before the column existed - is shown as "estimated" rather than
# hidden: no tag would read as "no concerns", which is the wrong way to be wrong.
CONFIDENCE_TAGS = {
    "verified_primary_source": ("Verified source", "conf-verified"),
    "cross_referenced":        ("Cross-checked", "conf-cross"),
    "estimated":               ("Estimated - verify before use", "conf-est"),
}


def confidence_tag(value):
    """The data-confidence tag for a material card, as escaped HTML."""
    key = "" if is_na(value) else str(value).strip().lower()
    label, css_class = CONFIDENCE_TAGS.get(key, CONFIDENCE_TAGS["estimated"])
    return f'<span class="conf-tag {css_class}">{esc(label)}</span>'


# The free tier that actually binds is Groq's 200,000 tokens per day. It is the
# only budget we can measure, so it is what the bar represents.
DAILY_TOKEN_BUDGET = 200_000


def usage_gauge():
    """(fraction_used, plain_english) for today's AI budget, or None.

    Deliberately returns no token counts. "5,192 of 200,000 tokens across 4
    calls to groq" is precise and useless to someone who just wants to know
    whether they can keep asking questions.
    """
    try:
        import llm as _llm
        used = _llm.usage_since(24)
    except Exception:
        return None
    if not used:
        return (0.0, "Nothing used yet today")
    spent = used.get("groq", {}).get("total", 0) or sum(u["total"] for u in used.values())
    frac = min(1.0, spent / DAILY_TOKEN_BUDGET)
    if frac < 0.25:
        phrase = "Plenty left today"
    elif frac < 0.60:
        phrase = "About a third used today"
    elif frac < 0.85:
        phrase = "Over half used today"
    else:
        phrase = "Running low today"
    return (frac, phrase)


def usage_bar_html(frac, phrase):
    """A slim bar. Colour shifts only when it is worth noticing."""
    pct = max(2, int(round(frac * 100)))     # a sliver is visible at 0
    colour = "var(--accent)"
    if frac >= 0.85:
        colour = "#B85C4C"
    elif frac >= 0.60:
        colour = "#C18A3E"
    return (
        '<div style="margin-top:0.15rem;">'
        '<div style="height:4px;border-radius:2px;background:rgba(38,38,36,0.09);'
        'overflow:hidden;">'
        f'<div style="width:{pct}%;height:100%;background:{colour};'
        'border-radius:2px;"></div></div>'
        f'<div style="font-size:0.72rem;color:var(--text-faint);margin-top:0.35rem;">'
        f'{phrase}</div></div>'
    )


def rank_badge(rank):
    """Rank chip. Only the top pick is accented - everything else is quiet."""
    cls = "rank-badge top" if rank == 1 else "rank-badge"
    return f'<span class="{cls}">{rank}</span>'


# How many materials to recommend. Was a sidebar slider; the reasoning prompt is
# tuned around three, and it is not a setting most users want to think about.
RESULTS_PER_ANSWER = 3


# =================================================================
# SESSION STATE
# =================================================================
if "conversations" not in st.session_state:
    # ChatGPT-style: a list of chats, each a sequence of message turns.
    # A message is {"role": "user", "content": str} or
    #              {"role": "assistant", "result": <reason_about_query dict>}.
    st.session_state.conversations = [{"title": "New chat", "messages": []}]
if "current" not in st.session_state:
    st.session_state.current = 0
if "editing_idx" not in st.session_state:
    st.session_state.editing_idx = None   # index of the message being edited inline


# =================================================================
# SIDEBAR
# =================================================================
with st.sidebar:
    st.markdown("#### Material Assistant")

    if st.button("New chat", use_container_width=True, type="primary"):
        convs = st.session_state.conversations
        st.session_state.editing_idx = None
        # Reuse the current chat if it's already empty, instead of stacking blanks.
        if convs[st.session_state.current]["messages"]:
            convs.append({"title": "New chat", "messages": []})
            st.session_state.current = len(convs) - 1
        st.rerun()

    st.markdown("---")
    st.caption("Chats")

    convs = st.session_state.conversations
    for i in range(len(convs) - 1, -1, -1):
        title = convs[i]["title"] or "New chat"
        label = (title[:30] + "…") if len(title) > 30 else title
        is_cur = (i == st.session_state.current)
        if st.button(
            label,
            key=f"conv_{i}",
            use_container_width=True,
            type="primary" if is_cur else "secondary",
        ):
            st.session_state.current = i
            st.session_state.editing_idx = None
            st.rerun()

    # Pinned at the bottom: how much of today's allowance is gone, as a bar.
    # Everything numeric moved to `py check_llm.py` - a sidebar is for glancing
    # at, not for reading.
    st.markdown("---")
    _gauge = usage_gauge()
    if _gauge is not None:
        st.markdown(
            '<div style="font-size:0.72rem;font-weight:600;letter-spacing:0.06em;'
            'text-transform:uppercase;color:var(--text-faint);">Daily usage</div>'
            + usage_bar_html(*_gauge),
            unsafe_allow_html=True,
        )

    st.write("")
    if st.button("Clear all chats", use_container_width=True, type="tertiary"):
        st.session_state.conversations = [{"title": "New chat", "messages": []}]
        st.session_state.current = 0
        st.rerun()


# =================================================================
# RENDER RESULT
# =================================================================
def render_result(result):
    # --- Small talk / off-topic: a plain reply, no material cards ---
    if result.get("chat_reply"):
        st.markdown(result["chat_reply"])
        return

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
        st.markdown(f'<div class="note note-accent">{esc(summary)}</div>', unsafe_allow_html=True)

    # --- Query understanding (collapsible) ---
    with st.expander("How your query was interpreted"):
        understood = result.get("understood", {})
        if result.get("resolved_query"):
            st.markdown(f"**Follow-up resolved to:** *{result['resolved_query']}*")
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
                    f'{rank_badge(rank)}<span class="mat-name">'
                    f'{html.escape(str(mat.get("common_name", "")))}</span>'
                    f'<div class="mat-meta">'
                    f'{html.escape(str(mat.get("material_class", "")).replace("_", " ").title())}'
                    f' &nbsp;·&nbsp; '
                    f'{html.escape(str(mat.get("condition", "")).replace("_", " "))}'
                    f' &nbsp;·&nbsp; <code>{html.escape(str(mat_id))}</code></div>',
                    unsafe_allow_html=True
                )
            with col_b:
                # 0-1 similarity, shown as a percentage because a bare decimal
                # reads like a score the user is meant to interpret.
                relevance = mat.get("relevance_score", 0) or 0
                st.markdown(
                    f'<div style="text-align:right;" class="muted">'
                    f'{relevance * 100:.0f}% match</div>',
                    unsafe_allow_html=True,
                )

            # Confidence tag before anything numeric, so the reader knows how
            # much weight the properties below can carry.
            st.markdown(confidence_tag(mat.get("data_confidence")),
                        unsafe_allow_html=True)

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
            pills_html = "".join(f'<span class="prop-pill">{esc(p)}</span>' for p in pills)
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
                    chips.append(f'<span class="prop-pill">{esc(method.replace("_", " "))}</span>')
            flam = mat.get("flammability")
            if not is_na(flam):
                fc = {"V-0": "#86ab8c", "V-1": "#9bb084", "V-2": "#b3ad80", "HB": "#b09280"}.get(str(flam), "#7c8088")
                chips.append(
                    f'<span class="prop-pill" style="border-color:{fc};color:{fc};">'
                    f'UL94 {esc(flam)}</span>'
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
                    f'<b>Allowable stress at {esc(stress_info.get("temperature_C"))}°C:</b> '
                    f'{esc(stress_info.get("stress_MPa"))} MPa{interp_note}<br>'
                    f'<span class="muted">Source: {esc(stress_info.get("source"))}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # LLM reasoning
            reasoning = result.get("reasoning", {}).get(mat_id, "")
            if reasoning:
                st.markdown(
                    f'<div class="note note-accent"><b>Why this fits:</b> {esc(reasoning)}</div>',
                    unsafe_allow_html=True
                )

            # Warnings
            warnings = mat.get("key_warnings", "")
            if not is_na(warnings):
                st.markdown(
                    f'<div class="warn"><b>Watch out for:</b> {esc(warnings)}</div>',
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


# =================================================================
# CHAT  (ChatGPT-style: message bubbles + input pinned at the bottom)
# =================================================================
# Clamp before indexing. A stale `current` (an older session, or any future
# delete-chat feature) would otherwise raise IndexError here and take the whole
# page down with a traceback instead of showing a chat.
if not st.session_state.conversations:
    st.session_state.conversations = [{"title": "New chat", "messages": []}]
st.session_state.current = max(
    0, min(st.session_state.current, len(st.session_state.conversations) - 1))
conv = st.session_state.conversations[st.session_state.current]

# Chat box is always pinned to the bottom regardless of where it's called.
prompt = st.chat_input("Describe your design conditions…")
if prompt and prompt.strip():
    conv["messages"].append({
        "role": "user", "content": prompt.strip(),
        "time": datetime.now().strftime("%H:%M"),
    })
    if not conv["title"] or conv["title"] == "New chat":
        conv["title"] = prompt.strip()

if not conv["messages"]:
    # Centered empty-state greeting, like a fresh ChatGPT thread.
    st.markdown(
        "<div style='text-align:center;margin-top:14vh;'>"
        "<div style='font-size:1.65rem;font-weight:600;color:var(--text);"
        "letter-spacing:-0.02em;'>Material Selection Assistant</div>"
        "<div style='margin-top:0.7rem;font-size:0.95rem;color:var(--text-muted);"
        "max-width:30rem;margin-left:auto;margin-right:auto;line-height:1.6;'>"
        "Describe what you are designing and the conditions it has to survive. "
        "Every recommendation comes from a verified property database."
        "</div>"
        "<div style='margin-top:1.6rem;font-size:0.78rem;color:var(--text-faint);'>"
        "AI-assisted recommendations — verify critical designs with an engineer."
        "</div></div>",
        unsafe_allow_html=True,
    )
    # Starter prompts: a blank chat box is the hardest thing to answer.
    st.write("")
    ex_cols = st.columns(3)
    EXAMPLES = [
        "A boat fitting that will not corrode in seawater",
        "A cheap plastic housing that sits outdoors in the sun",
        "A shaft carrying 300 MPa at 200 C",
    ]
    for col, example in zip(ex_cols, EXAMPLES):
        with col:
            if st.button(example, key=f"ex_{example[:12]}", use_container_width=True):
                conv["messages"].append({
                    "role": "user", "content": example,
                    "time": datetime.now().strftime("%H:%M"),
                })
                conv["title"] = example
                st.rerun()
else:
    # Replay the conversation: user messages as right-aligned WhatsApp-style
    # bubbles; under each message an inline action bar (regenerate / edit / copy)
    # next to the send time, like a standard AI chat.
    def user_bubble(text):
        st.markdown(
            f'<div class="wa-row user"><div class="wa-bubble-user">'
            f'{html.escape(text)}</div></div>',
            unsafe_allow_html=True,
        )

    def action_bar(idx, *, is_user, time_str):
        # Right-aligned cluster: send time + small borderless icon buttons.
        _, area = st.columns([0.44, 0.56])
        with area:
            if is_user:
                tc, b_re, b_ed, b_cp = st.columns([0.40, 0.20, 0.20, 0.20])
            else:
                tc, b_re, b_cp = st.columns([0.60, 0.20, 0.20])
            with tc:
                if time_str:
                    st.caption(time_str)
            with b_re:
                if st.button("Retry", key=f"regen_{idx}", type="tertiary", help="Regenerate this answer"):
                    # Keep this turn's user query, drop its answer + everything after.
                    conv["messages"] = conv["messages"][: (idx + 1 if is_user else idx)]
                    st.session_state.editing_idx = None
                    st.rerun()
            if is_user:
                with b_ed:
                    if st.button("Edit", key=f"edit_{idx}", type="tertiary", help="Edit this message"):
                        st.session_state.editing_idx = idx
                        st.rerun()
            with b_cp:
                if st.button("Copy", key=f"copy_{idx}", type="tertiary", help="Copy to clipboard"):
                    st.session_state["_pending_copy"] = (
                        conv["messages"][idx]["content"] if is_user
                        else conv["messages"][idx]["result"].get("summary", "")
                    )

    for i, msg in enumerate(conv["messages"]):
        if msg["role"] == "user":
            if st.session_state.editing_idx == i:
                # Inline edit mode (replaces the bubble while editing).
                edited = st.text_area("Edit message", value=msg["content"], key=f"editbox_{i}")
                save_c, cancel_c, _ = st.columns([0.2, 0.2, 0.6])
                with save_c:
                    if st.button("Save", key=f"save_{i}", type="primary", use_container_width=True):
                        nt = edited.strip()
                        if nt:
                            conv["messages"] = conv["messages"][:i] + [{
                                "role": "user", "content": nt,
                                "time": datetime.now().strftime("%H:%M"),
                            }]
                            if i == 0:
                                conv["title"] = nt
                        st.session_state.editing_idx = None
                        st.rerun()
                with cancel_c:
                    if st.button("Cancel", key=f"cancel_{i}", use_container_width=True):
                        st.session_state.editing_idx = None
                        st.rerun()
            else:
                user_bubble(msg["content"])
                action_bar(i, is_user=True, time_str=msg.get("time", ""))
        else:
            with st.chat_message("assistant", avatar="🔧"):
                render_result(msg["result"])
            action_bar(i, is_user=False, time_str=msg.get("time", ""))

    # One-shot client-side clipboard copy when a 📋 button was clicked.
    pending_copy = st.session_state.pop("_pending_copy", None)
    if pending_copy:
        payload = json.dumps(pending_copy)
        components.html(
            "<script>const t=" + payload + ";"
            "(navigator.clipboard&&navigator.clipboard.writeText)"
            "?navigator.clipboard.writeText(t).catch(()=>{})"
            ":(()=>{const a=document.createElement('textarea');a.value=t;"
            "document.body.appendChild(a);a.select();try{document.execCommand('copy')}"
            "catch(e){}a.remove();})();</script>",
            height=0,
        )
        st.toast("Copied to clipboard", icon="📋")

    # If the last turn is an unanswered user message, generate the answer now.
    if conv["messages"][-1]["role"] == "user":
        # Prior turns let the pipeline resolve follow-ups ("make it cheaper").
        prior = []
        pending_q = None
        for m in conv["messages"][:-1]:
            if m["role"] == "user":
                pending_q = m["content"]
            elif pending_q is not None:
                res = m.get("result", {})
                # Chat replies are kept too: "so you can't do X?" only makes sense
                # if the model can see the refusal it is responding to.
                prior.append({
                    "query": pending_q,
                    "materials": [x.get("common_name", "") for x in res.get("materials", [])],
                    "reply": res.get("chat_reply") or "",
                })
                pending_q = None

        with st.chat_message("assistant", avatar="🔧"):
            # Named stages instead of one opaque spinner: the pipeline makes two
            # LLM calls and a vector search, and saying which one is running makes
            # the wait feel like progress rather than a hang.
            with st.status("Working...", expanded=False) as status:
                result = reason_about_query(
                    conv["messages"][-1]["content"], top_k=RESULTS_PER_ANSWER, history=prior,
                    on_step=lambda label: status.update(label=label),
                )
                status.update(label="Done", state="complete")
        conv["messages"].append({
            "role": "assistant", "result": result,
            "time": datetime.now().strftime("%H:%M"),
        })
        st.rerun()