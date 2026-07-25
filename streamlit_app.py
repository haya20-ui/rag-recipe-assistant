import importlib.util
import os

import streamlit as st

st.set_page_config(
    page_title="Recipe RAG Assistant", page_icon="🍳", layout="centered"
)


def _load_module(filename, module_name):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Automatically build Chroma DB if missing
def ensure_chroma_db():
    if not os.path.exists("./chroma_db"):
        with st.spinner(
            "Building vector database for the first time... (this takes ~30 seconds)"
        ):
            _load_module("05_create_chroma_store.py", "builder")
            st.success("Database created successfully!")


st.title("🍳 Recipe RAG Assistant")
st.write(
    "Ask a cooking question and get an answer grounded in a local recipe database, with citations to the recipes used."
)

# Auto-check on load
ensure_chroma_db()

with st.form("rag_form"):
    user_query = st.text_input(
        "What would you like to cook?",
        placeholder="e.g. What can I make with chicken and rice?",
    )
    top_k = st.slider("Number of recipes to retrieve", 1, 5, 3)
    submitted = st.form_submit_button("Get Answer")

if submitted:
    if not user_query.strip():
        st.warning("Please enter a cooking question first.")
    else:
        # Double check DB exists
        ensure_chroma_db()

        retriever = _load_module("06_retrieve_context.py", "retriever")
        rag = _load_module("07_prompting.py", "rag")

        with st.spinner("Searching for relevant recipes..."):
            context_chunks = retriever.retrieve_relevant_chunks(
                user_query, top_k=top_k
            )

        if context_chunks is not None:
            if not context_chunks:
                st.info("No relevant recipes were found for your query.")
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
                        st.markdown(f"**{i}. {title}** ({source})")
                        if distance is not None:
                            st.caption(f"Distance score: {distance:.4f}")
                        st.text(chunk.get("text", ""))
                        st.divider()