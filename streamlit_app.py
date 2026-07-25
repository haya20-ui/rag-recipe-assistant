"""
streamlit_app.py
 
Streamlit front-end for the Recipe RAG pipeline. Lets the user ask a
natural-language cooking question, retrieves relevant recipe context
from the local ChromaDB store, generates a grounded answer via
07_prompting.py, and displays the answer along with an expandable view
of the cited recipe sources.
"""

import importlib.util
import os

import streamlit as st


def _load_module(module_filename: str, module_alias: str):
    """Dynamically load a .py file whose name is not a valid Python identifier."""
    module_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), module_filename
    )
    spec = importlib.util.spec_from_file_location(module_alias, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import 06_retrieve_context.py as `retriever` and 07_prompting.py as `rag`.
retriever = _load_module("06_retrieve_context.py", "retriever")
rag = _load_module("07_prompting.py", "rag")


# --- Required secrets check block ---
try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
        rag.OPENROUTER_MODEL = st.secrets.get(
            "OPENROUTER_MODEL", rag.OPENROUTER_MODEL
        )
except Exception:
    pass
# --- End required secrets check block ---


st.set_page_config(
    page_title="Recipe RAG Assistant", page_icon="🍳", layout="centered"
)

st.title("🍳 Recipe RAG Assistant")
st.write(
    "Ask a cooking question and get an answer grounded in a local recipe "
    "database, with citations to the recipes used."
)

if not os.path.exists("./chroma_db"):
    with st.spinner("Building vector database for the first time..."):
        _load_module("05_create_chroma_store.py", "builder")

# --- WRAPPED IN FORM TO FIX TYPING/PASTING ISSUES ---
with st.form(key="recipe_query_form"):
    user_query = st.text_input(
        "What would you like to cook?",
        placeholder="e.g. What can I make with chicken and rice?",
    )
    n_results = st.slider(
        "Number of recipes to retrieve", min_value=1, max_value=10, value=3
    )
    submitted = st.form_submit_button("Get Answer", type="primary")

if submitted:
    if not user_query or not user_query.strip():
        st.error("Please enter a question first.")
    else:
        with st.spinner("Retrieving relevant recipes..."):
            try:
                context_chunks = retriever.query_recipes(
                    user_query, n_results=n_results
                )
            except FileNotFoundError as e:
                st.error(str(e))
                context_chunks = None

        if context_chunks is not None:
            if not context_chunks:
                st.info("No relevant recipes were found for your question.")
            else:
                with st.spinner("Generating answer..."):
                    prompt = rag.build_prompt(user_query, context_chunks)
                    answer = rag.generate_response(prompt)

                st.subheader("Answer")
                st.write(answer)

                with st.expander("📚 Cited Sources / Retrieved Context"):
                    for i, chunk in enumerate(context_chunks, start=1):
                        title = chunk.get("metadata", {}).get(
                            "title", "Unknown Recipe"
                        )
                        source = chunk.get("metadata", {}).get(
                            "source", "Unknown Source"
                        )
                        distance = chunk.get("distance")
                        st.markdown(f"**{i}. {title}**")
                        st.markdown(f"Source: {source}")
                        if distance is not None:
                            st.caption(
                                f"Similarity distance: {distance:.4f}"
                            )
                        st.text(chunk.get("document", ""))
                        st.divider()