# app.py
# Streamlit web UI with chat history (display only)

import streamlit as st
from reason import reason_about_query

# -----------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Material Selection Assistant",
    page_icon="🔧",
    layout="wide",
)

# -----------------------------------------------------------------
# INITIALIZE SESSION STATE (the memory)
# -----------------------------------------------------------------
# This block only runs once per browser session
if "history" not in st.session_state:
    st.session_state.history = []   # list of {"query": "...", "answer": "..."}

# -----------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------
st.title("🔧 Mechanical Material Selection Assistant")
st.markdown(
    "Describe your design conditions in plain English. "
    "Get back ranked material recommendations with citations from "
    "Shigley, ASME, Cardarelli, Outokumpu, ESAB, and Machinery's Handbook."
)

# -----------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------
with st.sidebar:
    st.header("About this tool")
    st.markdown(
        "**Built for:** Mechanical engineering students and interns\n\n"
        "**How it works:**\n"
        "1. Your query goes to Gemini\n"
        "2. Gemini extracts structured filters\n"
        "3. ChromaDB + SQLite return matching materials\n"
        "4. Gemini writes the recommendation using ONLY database data\n\n"
        "Every property value traces to a verified source."
    )
    
    st.markdown("---")
    st.subheader("Example queries")
    st.markdown(
        "- *Lightweight material for marine use*\n"
        "- *Shaft at 300 MPa stress and 200°C*\n"
        "- *Food processing tank material*\n"
        "- *Cheap weldable steel for a bracket*\n"
        "- *Aerospace wing skin material*"
    )
    
    st.markdown("---")
    st.subheader("Database coverage (v1)")
    st.markdown(
        "10 metals:\n"
        "- 3 Stainless steels (303, 304L, 316L)\n"
        "- 3 Carbon steels (1018, 1040, 1050)\n"
        "- 2 Alloy steels (4130, 4140)\n"
        "- 2 Aluminum alloys (2024-T3, 5052-H32)"
    )
    
    st.markdown("---")
    st.subheader("Session")
    st.markdown(f"**Queries asked:** {len(st.session_state.history)}")
    
    # Clear history button
    if st.button("Clear history", type="secondary"):
        st.session_state.history = []
        st.rerun()  # refresh the page

# -----------------------------------------------------------------
# DISPLAY PAST QUERIES (newest at top, oldest at bottom)
# -----------------------------------------------------------------
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Previous queries this session")
    
    # Iterate in reverse so newest appears first
    for i, entry in enumerate(reversed(st.session_state.history)):
        # Use an expander so old queries are collapsed by default
        query_num = len(st.session_state.history) - i
        with st.expander(f"Q{query_num}: {entry['query'][:80]}{'...' if len(entry['query']) > 80 else ''}"):
            st.markdown(f"**Your query:** {entry['query']}")
            st.markdown("---")
            st.markdown(entry['answer'])

# -----------------------------------------------------------------
# INPUT AREA — at the bottom
# -----------------------------------------------------------------
st.markdown("---")
st.subheader("Ask a new question")

user_query = st.text_area(
    "Describe your design conditions:",
    placeholder="e.g., I need a material for a marine pump shaft that must handle 250 MPa at 150°C",
    height=100,
    key="query_input",
)

top_k = st.slider("Number of materials to recommend", min_value=2, max_value=5, value=3)

submit = st.button("Find Materials", type="primary")

# -----------------------------------------------------------------
# RUN PIPELINE WHEN SUBMITTED
# -----------------------------------------------------------------
if submit:
    if not user_query.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Thinking… (this may take 5–15 seconds)"):
            try:
                answer = reason_about_query(user_query, top_k=top_k)
                
                # Save to history
                st.session_state.history.append({
                    "query": user_query,
                    "answer": answer,
                })
                
                # Show the fresh answer immediately at the bottom
                st.markdown("---")
                st.subheader("📋 Result")
                st.markdown(answer)
                
                st.markdown("---")
                st.caption(
                    "Recommendations are based on a curated database of mechanical "
                    "engineering references. For critical applications, validate "
                    "with a qualified engineer."
                )
                
                # Hint to scroll up
                st.info("💡 This query has been added to your history above.")
                
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.info(
                    "If Gemini is busy, please try again in a minute. "
                    "If the error persists, check that your API key is set in .env."
                )