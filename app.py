# app.py
# Streamlit web UI for the Material Selection Assistant

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
# HEADER
# -----------------------------------------------------------------
st.title("🔧 Mechanical Material Selection Assistant")
st.markdown(
    "Describe your design conditions in plain English. "
    "Get back ranked material recommendations with citations from "
    "Shigley, ASME, Cardarelli, Outokumpu, ESAB, and Machinery's Handbook."
)

# -----------------------------------------------------------------
# SIDEBAR — info and example queries
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

# -----------------------------------------------------------------
# MAIN — input + results
# -----------------------------------------------------------------

# Query input
user_query = st.text_area(
    "Describe your design conditions:",
    placeholder="e.g., I need a material for a marine pump shaft that must handle 250 MPa at 150°C",
    height=100,
)

# Number of results selector
top_k = st.slider("Number of materials to recommend", min_value=2, max_value=5, value=3)

# Submit button
submit = st.button("Find Materials", type="primary")

# -----------------------------------------------------------------
# Run the pipeline when submitted
# -----------------------------------------------------------------
if submit:
    if not user_query.strip():
        st.warning("Please enter a query first.")
    else:
        # Show a spinner while the pipeline runs
        with st.spinner("Thinking… (this may take 5–15 seconds)"):
            try:
                answer = reason_about_query(user_query, top_k=top_k)
                
                # Display the answer
                st.markdown("---")
                st.markdown(answer)
                
                # Footer note
                st.markdown("---")
                st.caption(
                    "Recommendations are based on a curated database of mechanical "
                    "engineering references. For critical applications, validate "
                    "with a qualified engineer."
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.info(
                    "If Gemini is busy, please try again in a minute. "
                    "If the error persists, check that your API key is set in .env."
                )