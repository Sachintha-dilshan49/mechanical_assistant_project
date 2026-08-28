# reason.py
# End-to-end RAG pipeline with graceful fallback when the LLM is unavailable.
# Returns structured data for the UI to render as cards.

import json
import re

import llm
from understand_query import understand_query
from retrieve import find_candidates, get_allowable_stress

# -----------------------------------------------------------------
# REASONING PROMPT (LLM only writes summary + per-material reasoning)
# -----------------------------------------------------------------
REASONING_PROMPT_TEMPLATE = """You are a senior mechanical engineer helping a student choose a material.
You are given a POOL of candidate materials retrieved from a database. Your job is to SELECT and
RANK the ones that genuinely fit the application — NOT to repeat the retrieval order. The pool was
gathered broadly and WILL contain unsuitable options; filtering them out by reasoning is your job.

The student asked: "{user_query}"
Interpreted as: {semantic_query}

CRITICAL RULES:
1. All specific property VALUES and numbers come ONLY from the data below — never invent a number
   and never cite a source you were not given. The database is the source of truth for facts.
2. Apply real engineering judgment and physical common sense to pick the best materials for THIS
   job — e.g. low density floats while dense ferrous metals sink and rust in water; brittle
   materials shatter under impact; high hardness resists wear; car bodies are normally formable
   sheet steel or aluminum; cheap toys are usually plastic. Reasoning from principles is required;
   fabricating data is not.
3. SELECT the best {max_select} or fewer materials, ranked best-first, and ignore poor fits. If NONE
   of the pool is well suited, still pick the closest 1-2, say clearly they are compromises, and
   describe what property profile would actually suit the job.
4. NEVER invent a value. Each block lists ONLY the properties that apply to that material's
   family - a field that is absent is "not applicable" or unknown, never zero.

INTERPRET PROPERTIES BY FAMILY (material_class):
- METALS: corrosion ratings, weldability, machinability.
- PLASTICS (plastic_*): hardness_shore_d (NOT hardness_HB), max_continuous_use_temp_C (NOT
  max_service_temp_C), chemical_resistance_* (NOT corrosion_*); mention flammability/uv_resistance/
  water_absorption when relevant.
- CERAMICS (ceramic_*): no yield point (NOT_FOUND) — UTS is the flexural strength; ALWAYS warn it is
  brittle and must be designed for compression.
- COMPOSITES (composite_*): properties are directional (especially composite_cfrp); for cfrp note
  that galvanic isolation from aluminium is required.

{coverage_note}

CANDIDATE POOL:
{materials_block}

{stress_block}

Output STRICT JSON only (no markdown fences, no other text):
{{
    "overall_summary": "<2-3 sentences: the recommendation and the key tradeoff>",
    "selected": [
        {{"material_id": "<id from the pool above>", "reasoning": "<1-2 sentences why it fits this job>"}}
    ]
}}
The "selected" list must be ranked best-first and contain ONLY material_id values that appear in the
pool above. Order matters — the first entry is your #1 recommendation."""


# -----------------------------------------------------------------
# DATABASE COVERAGE
# The pool can only ever contain what the curated database holds. Without saying
# so, the model happily answers "rubber O-ring" with a cast epoxy and sounds
# certain about it. Naming the gap is the difference between a useful tool and a
# confidently wrong one.
# -----------------------------------------------------------------
_ABSENT_FAMILIES = ("elastomers / rubber (NBR, EPDM, silicone, natural rubber)",
                    "wood and engineered timber",
                    "concrete, masonry and stone",
                    "foams (EPS, PU, structural foam)",
                    "spring tempers and music wire",
                    "printable filament grades (PLA, PETG)")

_coverage_cache = None


def coverage_note() -> str:
    """A statement of what the database does and does not contain.

    Built from the live class list so it stays true as materials are added.
    """
    global _coverage_cache
    if _coverage_cache is not None:
        return _coverage_cache
    try:
        import db
        conn = db.connect()
        classes = sorted({r["material_class"] for r in db.fetch_materials(conn)})
        total = db.count_materials(conn)
        conn.close()
    except Exception:
        _coverage_cache = ""
        return _coverage_cache

    absent = [f for f in _ABSENT_FAMILIES
              if not any(f.split()[0].lower().strip(",") in c for c in classes)]
    _coverage_cache = (
        f"DATABASE COVERAGE ({total} materials). The pool above is everything available; "
        f"there is nothing else to fall back on.\n"
        f"Classes held: {', '.join(classes)}.\n"
        f"NOT in the database: {'; '.join(absent)}.\n"
        f"If this application genuinely needs a material family that is NOT held, say so "
        f"PLAINLY in the first sentence of overall_summary, name the family that would "
        f"actually suit the job, and present the listed candidates as the nearest available "
        f"options rather than as correct answers. Never imply a poor substitute is suitable."
    )
    return _coverage_cache


def _num(m, key):
    """Compact string for a numeric field, or None when it does not apply."""
    v = m.get(key)
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.upper() == "NOT_FOUND":
        return None
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else f"{f:g}"
    except ValueError:
        return s


def _txt(m, key, limit):
    """Trimmed text field, or None. Long prose is truncated to save tokens."""
    v = m.get(key)
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.upper() == "NOT_FOUND":
        return None
    return (s[:limit].rstrip() + "...") if len(s) > limit else s


def _join(pairs):
    """'a=1 b=2', skipping anything missing."""
    return " ".join(f"{k}={v}" for k, v in pairs if v)


def format_material_for_prompt(m: dict) -> str:
    """One candidate material as a compact text block for the LLM.

    Only the fields that mean something for this material's family are emitted:
    sending corrosion ratings for a plastic or Shore D for a steel is noise that
    costs tokens. That matters directly - the free-tier budget is 8k tokens per
    minute per model, and a pool of 18 verbose blocks alone used to exceed it,
    so every query got throttled and took ~50s.

    Missing values are simply omitted; the LLM is told that an absent field means
    "not applicable", which replaces the old NOT_FOUND spam.
    """
    cls = str(m.get("material_class", ""))
    is_plastic = cls.startswith("plastic_")
    is_ceramic = cls.startswith("ceramic_")
    is_composite = cls.startswith("composite_")
    is_metal = not (is_plastic or is_ceramic or is_composite)

    lines = [f"--- {m.get('common_name')} [{m.get('material_id')}] {cls}"]

    # Mechanical + physical: shared by every family.
    mech = [("ys_MPa", _num(m, "yield_strength_MPa")),
            ("uts_MPa", _num(m, "ultimate_tensile_strength_MPa")),
            ("elong%", _num(m, "elongation_percent")),
            ("E_GPa", _num(m, "elastic_modulus_GPa")),
            ("rho", _num(m, "density_kg_m3"))]
    if is_ceramic:
        # No yield point; UTS is the flexural strength.
        mech = [("flexural_MPa", _num(m, "ultimate_tensile_strength_MPa"))] + mech[2:]
    lines.append("  " + _join(mech))

    # Temperature: plastics are limited by continuous-use temp, others by service temp.
    temps = [("Tmax_C", _num(m, "max_service_temp_C")),
             ("Tmin_C", _num(m, "min_service_temp_C"))]
    if is_plastic:
        temps.insert(0, ("Tcont_C", _num(m, "max_continuous_use_temp_C")))
    lines.append("  " + _join(temps))

    if is_metal:
        corr = "/".join(_num(m, k) or "-" for k in
                        ("corrosion_seawater", "corrosion_acidic", "corrosion_alkaline",
                         "corrosion_atmospheric", "corrosion_high_temp"))
        lines.append(f"  corrosion sea/acid/alk/atmos/hot(1-5)={corr}  " +
                     _join([("HB", _num(m, "hardness_HB")),
                            ("weld1_5", _num(m, "weldability")),
                            ("machinability", _num(m, "machinability_index"))]))
    else:
        chem = "/".join(_num(m, k) or "-" for k in
                        ("chemical_resistance_solvents", "chemical_resistance_acids",
                         "chemical_resistance_alkalis", "chemical_resistance_fuels"))
        extras = [("shoreD", _num(m, "hardness_shore_d")),
                  ("UL94", _txt(m, "flammability", 8)),
                  ("uv1_5", _num(m, "uv_resistance")),
                  ("water_abs%", _num(m, "water_absorption_percent"))]
        lines.append(f"  chem_resist solv/acid/alk/fuel(1-5)={chem}  " + _join(extras))

    lines.append("  " + _join([("cost1_5", _num(m, "cost_class")),
                               ("usd_per_kg", _txt(m, "approx_cost_usd_per_kg", 12)),
                               ("fatigue1_5", _num(m, "fatigue_rating")),
                               ("join", _txt(m, "joining_method", 40))]))

    apps = _txt(m, "typical_applications", 90)
    if apps:
        lines.append(f"  uses: {apps}")
    warn = _txt(m, "key_warnings", 110)
    if warn:
        lines.append(f"  warn: {warn}")

    return "\n".join(l for l in lines if l.strip())


# -----------------------------------------------------------------
# REASONING CALL
# Model choice, provider failover (Groq -> Gemini), transient retries and quota
# cooldowns all live in llm.generate.
# -----------------------------------------------------------------
def call_llm_with_fallback(prompt: str):
    """Returns (text, warning). text is None when every provider failed."""
    return llm.generate(prompt, task="smart", json_output=True)


def _as_num(val):
    """Coerce a metadata value to float, or None if missing / NOT_FOUND.
    Lets the rule-based fallback compare values without crashing on the
    'NOT_FOUND' sentinel that non-applicable fields now carry."""
    try:
        if val is None:
            return None
        s = str(val).strip()
        if s == "" or s.upper() == "NOT_FOUND":
            return None
        return float(s)
    except (TypeError, ValueError):
        return None


def build_fallback_summary(materials: list, query: str) -> dict:
    """
    Build a basic summary from retrieval data alone, without LLM.
    Used when Gemini is completely unavailable. Handles metals and non-metals,
    and is safe against NOT_FOUND values.
    """
    if not materials:
        return {"overall_summary": "", "material_reasoning": {}}

    top = materials[0]
    # This runs when NO model was reachable, so nothing has judged whether these
    # materials actually suit the job - they are just the closest text matches.
    # Saying "materials matching your query" here would be a straight lie, and it
    # is exactly how a foam query ends up recommending Kevlar composite.
    summary = (
        f"**AI reasoning is unavailable right now, so these are NOT vetted "
        f"recommendations.** Listed below are the {len(materials)} closest matches by "
        f"text similarity only - nothing has checked that they suit your application. "
        f"The nearest match was {top.get('common_name')}. Please compare the property "
        f"cards yourself, and re-run the query later for a proper recommendation."
    )

    reasoning = {}
    for mat in materials:
        cls = str(mat.get("material_class", ""))
        parts = []

        # metal-oriented signals
        if (_as_num(mat.get("corrosion_seawater")) or 0) >= 4:
            parts.append("good seawater resistance")
        if (_as_num(mat.get("weldability")) or 0) >= 4:
            parts.append("easily weldable")
        if (_as_num(mat.get("machinability_index")) or 0) >= 70:
            parts.append("highly machinable")

        # non-metal signals
        if (_as_num(mat.get("chemical_resistance_acids")) or 0) >= 4 \
                or (_as_num(mat.get("chemical_resistance_solvents")) or 0) >= 4:
            parts.append("strong chemical resistance")
        shore = _as_num(mat.get("hardness_shore_d"))
        if shore is not None and shore >= 75:
            parts.append("hard, wear-resistant surface")

        # shared strength / cost signals (UTS covers ceramics & composites)
        if (_as_num(mat.get("yield_strength_MPa")) or 0) >= 500 \
                or (_as_num(mat.get("ultimate_tensile_strength_MPa")) or 0) >= 800:
            parts.append("high strength")
        if (_as_num(mat.get("cost_class")) or 99) <= 1:
            parts.append("low cost")

        # family-specific safety notes
        if cls.startswith("ceramic_"):
            parts.append("brittle — design for compression")
        if cls == "composite_cfrp":
            parts.append("directional properties")

        if parts:
            reasoning[mat["material_id"]] = (
                f"Not assessed for your application. Notable properties: "
                f"{', '.join(parts)}. Check the warnings below before using it."
            )
        else:
            reasoning[mat["material_id"]] = (
                "Not assessed for your application - retrieved by text similarity "
                "only. Check the properties and warnings below."
            )

    return {"overall_summary": summary, "material_reasoning": reasoning}


# -----------------------------------------------------------------
# FOLLOW-UP RESOLUTION (make the chat conversation-aware)
# -----------------------------------------------------------------
CONTEXTUALIZE_PROMPT = """You rewrite a user's latest chat message into ONE standalone material-selection query.
Use the recent conversation to resolve references like "it", "that one", "cheaper", "instead",
"what about ...", "for saltwater". Keep the application/object from the conversation unless the user
clearly switches to a new topic.

Output ONLY the rewritten query on a single line — no quotes, no explanation. If the latest message
is already self-contained or starts a new unrelated topic, output it unchanged.

Recent conversation:
{history_block}

Latest message: {user_query}

Standalone query:"""


def contextualize_query(user_query: str, history: list) -> str:
    """Rewrite a follow-up into a self-contained query using recent turns.
    Best-effort: on any failure (or no history) returns the original message."""
    if not history:
        return user_query

    lines = []
    for turn in history[-3:]:                       # last few turns is enough context
        if turn.get("reply") and not turn.get("materials"):
            continue                                # a chat turn is not design context
        mats = ", ".join((turn.get("materials") or [])[:3])
        line = f'- User asked: "{turn.get("query", "")}"'
        if mats:
            line += f' -> recommended: {mats}'
        lines.append(line)

    prompt = CONTEXTUALIZE_PROMPT.format(history_block="\n".join(lines), user_query=user_query)
    try:
        raw, _warning = llm.generate(prompt, task="fast")
        text = (raw or "").strip().strip('"').strip()
        rewritten = text.splitlines()[0].strip() if text else ""
        return rewritten or user_query
    except Exception as e:
        print(f"  contextualize failed: {e}")
        return user_query


# -----------------------------------------------------------------
# INTENT GATE
# Not every message is a design question. Without this, "hey" runs the full
# pipeline and comes back with three material cards, which is nonsense.
# Matching is on the WHOLE normalised message, so "hey what material for a
# boat" is still treated as a real query.
# -----------------------------------------------------------------
GREETINGS = {
    "hi", "hey", "hello", "yo", "hiya", "howdy", "sup", "heya",
    "hey there", "hi there", "hello there", "good morning",
    "good afternoon", "good evening", "whats up", "wassup", "greetings",
}
THANKS = {"thanks", "thank you", "thx", "ty", "cheers", "appreciate it",
          "thanks a lot", "thank you so much", "nice", "great", "cool", "awesome"}
FAREWELLS = {"bye", "goodbye", "see you", "cya", "later", "good night", "gn"}
ACKS = {"ok", "okay", "k", "got it", "alright", "sure", "yes", "yeah", "yep",
        "no", "nope", "hmm", "hm", "idk", "test", "testing"}

# Questions ABOUT the tool rather than about a material. Anchored to the whole
# message so "what materials for a boat hull" stays a design question - a bare
# substring check on "what materials" would hijack it.
ABOUT_SYSTEM_PATTERNS = [
    r"^what (do |does |can )?(u|you|this|it|the system|the app) ?(have|has|hold|contain|offer|do|does)?$",
    r"^what (are|is) (these|this|that|it)( system| app| tool| thing| for)?$",
    r"^what(?:'s| is) (this|it)( about| for)?$",
    # These must END here. A trailing wildcard would swallow "what materials do
    # you have FOR MARINE USE", which is a design question, not a coverage one.
    r"^what (kind of |type of |types of )?(data|materials|classes|families)"
    r"( (do|does) (you|it|this|the system) (have|hold|contain|offer))?$",
    r"^what (materials|classes|families|data) are (there|available)$",
    r"^what (materials|classes|families) (are|is) in (your|the) (database|registry)$",
    r"^how many (materials|records|rows|things)( (do|does) (you|it|this) have| are there)?$",
    r"^(list|show)( me)?( all)?( the)? (materials|database|registry|classes)$",
    r"^what(?:'s| is) in (your|the) (database|registry|system)$",
    r"^what can (you do|i ask|this do|it do|i do (here|with this))$",
    r"^(who|what) (are|r) (you|u)$",
    r"^(how does (this|it) work|how do i use (this|it)|help|what do you do)$",
    r"^what (database|dataset|registry) (do you|does it) use$",
    r"^(give me |can you give me |tell me )?(a |an )?(intro|introduction|overview)"
    r"( about| of| on| to)? (you|u|this|it|yourself|the system|the tool)$",
    r"^(introduce|describe) (yourself|you|this|the system|the tool)$",
    r"^tell me about (you|u|this|it|yourself|the system|the tool)$",
    r"^what (do|does) (you|u|this|it) do$",
]

SMALL_TALK_REPLIES = {
    "thanks":   "You're welcome. Ask me about another design whenever you need to.",
    "farewell": "Goodbye - come back when you have another material to pick.",
}

EXAMPLES_BLOCK = (
    "Try asking:\n"
    "- *a shaft that carries 300 MPa at 200 C*\n"
    "- *a cheap plastic housing that sits outdoors in the sun*\n"
    "- *a boat fitting that will not corrode in seawater*\n\n"
    "The more you say about temperature, environment, loads and whether cost or "
    "weight matters, the better the recommendation."
)

# Below this average similarity over the top 3 hits, nothing in the database is
# even topically related. Measured separation: real material queries score
# 0.28-0.48, off-topic questions (maths, trivia, translation) score 0.05-0.11.
OFF_TOPIC_RELEVANCE = 0.18

_overview_cache = None


def system_overview() -> str:
    """Describe the tool using the live registry, not a hardcoded blurb.

    Answers "what do you have?" with what is actually in the database, so it
    stays correct as materials are added - and needs no API call, which means it
    still works when the LLM quota is exhausted.
    """
    global _overview_cache
    if _overview_cache is not None:
        return _overview_cache
    try:
        import db
        conn = db.connect()
        rows = db.fetch_materials(conn)
        stress_tables = conn.execute(
            "SELECT COUNT(DISTINCT stress_table_id) FROM allowable_stress").fetchone()[0]
        conn.close()
    except Exception:
        _overview_cache = ("I'm a material selection assistant.\n\n" + EXAMPLES_BLOCK)
        return _overview_cache

    families = {"Metals": [], "Plastics": [], "Ceramics": [], "Composites": []}
    for r in rows:
        cls = str(r.get("material_class", ""))
        if cls.startswith("plastic_"):
            families["Plastics"].append(r)
        elif cls.startswith("ceramic_"):
            families["Ceramics"].append(r)
        elif cls.startswith("composite_"):
            families["Composites"].append(r)
        else:
            families["Metals"].append(r)

    lines = [
        f"I'm a material selection assistant. I recommend engineering materials from a "
        f"curated database of **{len(rows)} materials** across "
        f"{len({r['material_class'] for r in rows})} classes - every value is from a "
        f"cited source, so I never invent numbers.\n",
        "**What's in the database**",
    ]
    for fam, items in families.items():
        if not items:
            continue
        names = sorted({str(i["material_class"]).replace("_", " ") for i in items})
        lines.append(f"- **{fam} ({len(items)})** - {', '.join(names)}")

    lines.append(
        f"\nEach material carries yield and tensile strength, density, stiffness, "
        f"temperature limits, corrosion or chemical-resistance ratings, weldability, "
        f"cost and typical uses. I also hold allowable-stress-vs-temperature tables "
        f"(ASME BPVC) for {stress_tables} materials."
    )
    lines.append(
        "\n**Not included:** rubber and elastomers, wood, concrete, foams, spring "
        "tempers, and 3D-printing filament grades. If you ask for those I'll tell you "
        "rather than suggest a poor substitute.\n"
    )
    lines.append(EXAMPLES_BLOCK)
    _overview_cache = "\n".join(lines)
    return _overview_cache


CHAT_PROMPT = """You are the conversational voice of a mechanical engineering material
selection tool. Reply to the user's latest message naturally, in 2-4 short sentences.

THE ONLY CAPABILITIES YOU MAY CLAIM:
{facts}

RULES:
- Never claim an ability that is not listed above. Never invent material data here.
- NEVER say you cannot help with materials or material questions - that is your whole
  purpose. Out of scope means subjects OTHER than materials (maths, coding, trivia).
- If the user asks for something out of scope, say so plainly and briefly, then point at
  what you CAN do. Do not lecture and do not apologise repeatedly.
- If they are reacting to a refusal you already gave (e.g. "so you can't do X?"), just
  confirm it and move the conversation forward. Do NOT repeat the same explanation again.
- Suggest an example query only when it genuinely helps, and at most two.
- Plain prose, no headings, no bullet lists unless you are listing what the database holds.
- Vary your wording between turns. Never paste the same paragraph twice.

Recent conversation:
{history}

User: {user_query}
Reply:"""


def conversational_reply(user_query, history=None, kind="about_system"):
    """An LLM-written reply for a non-material message, or None if no LLM.

    Grounded in system_overview() so the model can only describe capabilities the
    tool actually has - the same "database is the brain" rule the material path
    follows, applied to conversation. Falls back to the canned text offline.
    """
    lines = []
    for turn in (history or [])[-3:]:
        q = (turn.get("query") or "").strip()
        if q:
            lines.append(f"User: {q}")
        prev = (turn.get("reply") or "").strip()
        if prev:
            lines.append(f"You: {prev[:200]}")
        mats = ", ".join((turn.get("materials") or [])[:3])
        if mats:
            lines.append(f"You recommended: {mats}")
    history_block = "\n".join(lines) if lines else "(this is the first message)"

    prompt = CHAT_PROMPT.format(facts=system_overview(),
                                history=history_block,
                                user_query=user_query)
    try:
        text, _warning = llm.generate(prompt, task="fast")
    except Exception as exc:
        print(f"  conversational reply failed: {exc}")
        return None
    text = (text or "").strip()
    return text or None


def off_topic_reply(reason: str = "") -> str:
    """Decline a non-materials question, saying WHY and what this tool is for."""
    lead = reason.strip() if reason else (
        "That isn't a materials question, so I can't help with it")
    if not lead.endswith((".", "!", "?")):
        lead += "."
    return (f"{lead} I'm a material selection tool - I only recommend engineering "
            f"materials from a verified property database, so anything outside that "
            f"(maths, coding, general knowledge) is out of scope for me.\n\n"
            + EXAMPLES_BLOCK)


def _normalise(text):
    """Lowercase, drop punctuation and emoji, collapse whitespace."""
    cleaned = "".join(ch if (ch.isalnum() or ch.isspace() or ch == "'") else " "
                      for ch in (text or "").lower())
    return " ".join(cleaned.split())


def classify_local(user_query):
    """Intent from the message alone - no API call, so it works when quota is out.

    Returns "greeting" | "thanks" | "farewell" | "ack" | "about_system" | None.
    None means "might be a real material question" and the pipeline runs.
    """
    t = _normalise(user_query)
    if not t:
        return "greeting"
    for group, kind in ((GREETINGS, "greeting"), (THANKS, "thanks"),
                        (FAREWELLS, "farewell"), (ACKS, "ack")):
        if t in group:
            return kind
    for pattern in ABOUT_SYSTEM_PATTERNS:
        if re.match(pattern, t):
            return "about_system"
    return None


def small_talk_reply(user_query):
    """The canned answer for a non-material message, or None to run the pipeline."""
    kind = classify_local(user_query)
    if kind is None:
        return None
    if kind == "about_system":
        return system_overview()
    if kind in SMALL_TALK_REPLIES:
        return SMALL_TALK_REPLIES[kind]
    lead = "Hello. " if kind == "greeting" else "Ready when you are. "
    return lead + system_overview()


# -----------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------
def reason_about_query(user_query: str, top_k: int = 3, history: list = None,
                       on_step=None) -> dict:
    """
    Full RAG pipeline with graceful degradation.
    `history` (optional) is a list of prior turns [{"query": str, "materials": [names]}];
    when present, a follow-up like "make it cheaper" is resolved against it first.
    `on_step` (optional) is called with a short progress label as each stage
    starts, so the UI can say what it is doing instead of showing one long spinner.
    Always returns materials if any matched; LLM summary is best-effort.
    """
    def step(label):
        if on_step:
            on_step(label)
    result = {
        "query": user_query,
        "resolved_query": None,
        "understood": None,
        "materials": [],
        "stress_lookups": {},
        "stress_gaps": {},
        "summary": "",
        "reasoning": {},
        "warning": None,
        "error": None,
        "chat_reply": None,
    }

    # Step 0a: pure pleasantries are answered locally - they are formulaic, and
    # spending a model call on "thanks" is waste. Questions ABOUT the tool fall
    # through to the LLM below so the answer can respond to what was actually
    # asked; the local text is only used when no model is reachable.
    kind = classify_local(user_query)
    if kind in ("thanks", "farewell"):
        result["chat_reply"] = SMALL_TALK_REPLIES[kind]
        result["summary"] = result["chat_reply"]
        return result
    if kind in ("greeting", "ack", "about_system"):
        reply = conversational_reply(user_query, history, kind="about_system")
        if reply is None:
            reply = small_talk_reply(user_query)
        result["chat_reply"] = reply
        result["summary"] = reply
        return result

    # Step 0: resolve a follow-up against the conversation so "make it cheaper" /
    # "what about saltwater" inherit the earlier subject. No-op for a first message.
    if history:
        step("Reading the conversation so far...")
    search_query = contextualize_query(user_query, history) if history else user_query
    if search_query != user_query:
        result["resolved_query"] = search_query

    # Step 1: Understand query (with fallback)
    step("Understanding your requirements...")
    understood = None
    try:
        understood = understand_query(search_query)
    except Exception as e:
        print(f"  understand_query failed: {e}")
    
    llm_unavailable = understood is None
    if not understood:
        understood = {
            "semantic_query": search_query,
            "filters": {},
            "extracted_constraints": {},
            "reasoning": "Query understanding unavailable; using raw text for semantic search.",
        }
        result["warning"] = "Could not parse query structure - searched semantically only."

    # The word list catches the common cases; understand_query flags anything
    # else that is not a material question ("what is the weather in Paris").
    intent = str(understood.get("intent", "material_query")).lower()
    if intent in ("off_topic", "about_system"):
        # Let the model actually converse. Two different questions should not get
        # the same paragraph back, and a follow-up should answer the follow-up.
        reply = conversational_reply(user_query, history, kind=intent)
        if reply is None:
            reply = (system_overview() if intent == "about_system"
                     else off_topic_reply(understood.get("off_topic_reason", "")))
        result["chat_reply"] = reply
        result["summary"] = reply
        result["understood"] = understood
        return result

    result["understood"] = understood

    semantic_query = understood.get("semantic_query", search_query)
    filters = understood.get("filters", {})
    constraints = understood.get("extracted_constraints", {})

    family = constraints.get("material_family")

    # Step 2: Retrieve a BROAD candidate pool. Filters only gather candidates here;
    # the reasoning LLM makes the actual choice, so a too-tight filter can no longer
    # silently delete the right answer before the model ever sees it.
    POOL_SIZE = 18
    step("Searching the materials database...")
    pool = find_candidates(
        query_text=semantic_query,
        filters=filters if filters else None,
        family=family,
        pool_size=POOL_SIZE,
    )
    if not pool:
        result["error"] = "No materials in the database match this query."
        return result
    # Safety net: when no model was reachable, nothing has judged whether this is
    # even a materials question. Semantic similarity is the only signal left, and
    # it separates cleanly at this range, so use it rather than answering a maths
    # question with material cards.
    if llm_unavailable and pool:
        top3 = [m.get("relevance_score", 0.0) for m in pool[:3]]
        if sum(top3) / len(top3) < OFF_TOPIC_RELEVANCE:
            result["chat_reply"] = off_topic_reply(
                "That doesn't look like a materials question, and I can't check it "
                "properly right now because the AI service is unavailable")
            result["summary"] = result["chat_reply"]
            return result

    pool_by_id = {m["material_id"]: m for m in pool}

    # Step 3: Ask the LLM to SELECT and RANK the best materials from the pool.
    step(f"Comparing {len(pool)} candidate materials...")
    materials_block = "\n\n".join(format_material_for_prompt(m) for m in pool)
    final_prompt = REASONING_PROMPT_TEMPLATE.format(
        user_query=search_query,
        semantic_query=semantic_query,
        max_select=top_k,
        materials_block=materials_block,
        coverage_note=coverage_note(),
        stress_block="",
    )

    try:
        text, fallback_note = call_llm_with_fallback(final_prompt)
    except Exception as e:
        # Non-transient Gemini error — never crash the pipeline; fall back below.
        print(f"  LLM reasoning failed: {e}")
        text, fallback_note = None, "AI selection unavailable - reasoning service errored. Showing top database matches."

    selected = []
    reasoning = {}
    summary = ""

    if text:
        try:
            text = text.strip()
            if text.startswith("```"):
                text = "\n".join(l for l in text.split("\n") if not l.startswith("```"))
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("top-level JSON is not an object")
            summary = parsed.get("overall_summary") or ""
            if not isinstance(summary, str):
                summary = str(summary)
            # Everything below is defensive on purpose: this is model output, so
            # "selected" can be a string, a list of strings, or contain ids that
            # were never in the pool. None of that may reach the UI as an
            # exception - the pipeline has a working fallback right below.
            items = parsed.get("selected")
            if not isinstance(items, list):
                items = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                mid = item.get("material_id")
                if mid in pool_by_id and mid not in reasoning:   # ignore hallucinated/dupe ids
                    selected.append(pool_by_id[mid])
                    note = item.get("reasoning", "")
                    reasoning[mid] = note if isinstance(note, str) else str(note)
                if len(selected) >= top_k:
                    break
            if fallback_note:
                result["warning"] = fallback_note
        except (json.JSONDecodeError, ValueError, AttributeError, TypeError) as exc:
            print(f"  malformed model output ({type(exc).__name__}): {exc}")
            selected, reasoning, summary = [], {}, ""
            result["warning"] = "AI returned malformed output - showing top database matches."

    if not selected:
        # Fallback: take the best pool matches with rule-based reasoning.
        selected = pool[:top_k]
        fb = build_fallback_summary(selected, search_query)
        summary = summary or fb["overall_summary"]
        reasoning = fb["material_reasoning"]
        if not result["warning"]:
            result["warning"] = fallback_note or "AI summary unavailable. Showing top database matches."

    result["materials"] = selected
    result["summary"] = summary
    result["reasoning"] = reasoning

    # Step 4: Allowable-stress lookups for the SELECTED materials (database only).
    # Only 7 of the 66 materials have a stress table, so a miss is the normal
    # case, not the exception. Record WHY each miss happened in stress_gaps:
    # a card that just omits the stress row reads as "checked, nothing to
    # report", and the student has no way to tell that apart from a real pass.
    temp_C = constraints.get("temperature_C")
    if temp_C is not None:
        step("Looking up allowable stress at temperature...")
        for mat in selected:
            # NOT_FOUND, not "", is what a material without a table carries here.
            stress_table_id = _txt(mat, "stress_table_id", 64)
            if not stress_table_id:
                result["stress_gaps"][mat["material_id"]] = {"reason": "no_table"}
                continue
            stress_info = get_allowable_stress(stress_table_id, float(temp_C))
            if stress_info and stress_info.get("stress_MPa") is not None:
                result["stress_lookups"][mat["material_id"]] = {
                    "stress_MPa": stress_info["stress_MPa"],
                    "source": stress_info["source"],
                    "interpolated": stress_info.get("interpolated", False),
                    "temperature_C": temp_C,
                }
            elif stress_info and stress_info.get("error"):
                # The table exists but stops short of this temperature - a
                # different fact from having no table at all, and a more
                # alarming one, so it is passed through verbatim.
                result["stress_gaps"][mat["material_id"]] = {
                    "reason": "out_of_range",
                    "error": stress_info["error"],
                }
            else:
                # Table id references nothing in allowable_stress (dangling).
                result["stress_gaps"][mat["material_id"]] = {"reason": "no_table"}

    return result


# -----------------------------------------------------------------
# DEMO (run from terminal)
# -----------------------------------------------------------------
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    for q in [
        "lightweight material for marine use",
        "low-friction plastic gear that resists fuels",
        "brittle electrical insulator for 1000 C",
    ]:
        print("\n" + "=" * 70)
        print(f"QUERY: {q}")
        print("=" * 70)
        out = reason_about_query(q, top_k=3)
        print(json.dumps({k: v for k, v in out.items() if k != "materials"}, indent=2))
        print(f"Materials returned: {len(out['materials'])} -> "
              f"{[m.get('common_name') for m in out['materials']]}")