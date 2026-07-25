"""
06_retrieve_context.py

Provides `query_recipes()` for retrieving the most relevant recipe
chunks from the persistent ChromaDB collection built by
05_create_chroma_store.py.

No API keys are required or used in this module (embeddings are computed
locally with Sentence-Transformers, matching the store's embedding function).
"""

import importlib.util
import os

import chromadb


def _load_module(module_filename: str, module_alias: str):
    """Dynamically load a .py file whose name is not a valid Python identifier."""
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), module_filename)
    spec = importlib.util.spec_from_file_location(module_alias, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vector_module = _load_module("04_vector_representation.py", "vector_module_retrieve")
get_embedding_function = vector_module.get_embedding_function


CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "recipes_collection"

# Cache the client/collection so repeated calls (e.g. from Streamlit)
# don't reopen the database every time.
_client = None
_collection = None


def _get_collection():
    global _client, _collection

    if _collection is not None:
        return _collection

    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(
            f"Chroma store not found at '{CHROMA_DB_PATH}'. "
            f"Run 'python 05_create_chroma_store.py' first to build it."
        )

    _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_function = get_embedding_function()

    _collection = _client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )
    return _collection


def query_recipes(user_query: str, n_results: int = 3):
    """
    Query the recipes_collection for the most relevant chunks.

    Args:
        user_query (str): The natural-language user question.
        n_results (int): Number of top matching chunks to return.

    Returns:
        list[dict]: A list of results, each shaped as:
            {
                "document": str,     # the recipe text block
                "text": str,         # alias of "document" (compatibility)
                "metadata": dict,    # {"title": ..., "source": ...}
                "distance": float,   # similarity distance (lower = closer)
            }
    """
    if not user_query or not user_query.strip():
        return []

    collection = _get_collection()

    results = collection.query(
        query_texts=[user_query],
        n_results=n_results,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(documents)

    retrieved = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        retrieved.append(
            {
                "document": doc,
                "text": doc,  # alias, since some callers expect "text"
                "metadata": meta,
                "distance": dist,
            }
        )

    return retrieved


def retrieve_relevant_chunks(user_query: str, top_k: int = 3):
    """
    Alias for query_recipes() matching the (user_query, top_k=...) call
    signature used by streamlit_app.py.

    Args:
        user_query (str): The natural-language user question.
        top_k (int): Number of top matching chunks to return.

    Returns:
        list[dict]: Same shape as query_recipes().
    """
    return query_recipes(user_query, n_results=top_k)


if __name__ == "__main__":
    query = "What can I make with chicken and rice?"
    results = query_recipes(query, n_results=3)
    print(f"Query: {query}\n")
    for i, r in enumerate(results, start=1):
        print(f"Result {i}:")
        print(f"  Title: {r['metadata'].get('title')}")
        print(f"  Source: {r['metadata'].get('source')}")
        print(f"  Distance: {r['distance']}")
        print(f"  Document preview: {r['document'][:150]}...")
        print()