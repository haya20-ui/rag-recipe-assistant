"""
05_create_chroma_store.py
 
Orchestrates the full ingestion pipeline:
    01_documents -> 02_preprocessing -> 03_chunking -> 04_vector_representation
and writes the resulting embeddings into a persistent ChromaDB collection
named "recipes_collection" stored on disk at "./chroma_db".
 
Run this file directly to (re)build the vector store:
    python 05_create_chroma_store.py
 
No API keys are required or used in this module (embeddings are computed
locally with Sentence-Transformers).
 
NOTE ON IMPORTS:
Python module names cannot start with a digit (e.g. `import 01_documents`
is invalid syntax). Since the required file names start with digits
(01_documents.py, 02_preprocessing.py, etc.), this script loads them
dynamically using `importlib.util` instead of a normal `import` statement.
"""
 
import os
import importlib.util
 
import chromadb
 
 
def _load_module(module_filename: str, module_alias: str):
    """Dynamically load a .py file whose name is not a valid Python identifier."""
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), module_filename)
    spec = importlib.util.spec_from_file_location(module_alias, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
 
 
# Dynamically load the numbered pipeline modules.
documents_module = _load_module("01_documents.py", "documents_module")
preprocessing_module = _load_module("02_preprocessing.py", "preprocessing_module")
chunking_module = _load_module("03_chunking.py", "chunking_module")
vector_module = _load_module("04_vector_representation.py", "vector_module")
 
load_documents = documents_module.load_documents
clean_recipe_data = preprocessing_module.clean_recipe_data
create_recipe_chunks = chunking_module.create_recipe_chunks
get_embedding_function = vector_module.get_embedding_function
 
 
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "recipes_collection"
 
 
def build_vector_store(csv_path: str = "1_Recipe_csv.csv", sample_size: int = 300):
    """
    Run the full ingestion pipeline and persist embeddings to ChromaDB.
 
    Args:
        csv_path (str): Path to the recipe CSV file.
        sample_size (int): Max number of rows to sample from the CSV.
    """
    print("Step 1/5: Loading documents...")
    raw_recipes = load_documents(csv_path=csv_path, sample_size=sample_size)
    print(f"  -> Loaded {len(raw_recipes)} raw recipes.")
 
    print("Step 2/5: Cleaning and preprocessing...")
    cleaned_recipes = clean_recipe_data(raw_recipes)
    print(f"  -> {len(cleaned_recipes)} recipes remain after cleaning.")
 
    print("Step 3/5: Chunking recipes...")
    chunks = create_recipe_chunks(cleaned_recipes)
    print(f"  -> Created {len(chunks)} chunks.")
 
    print("Step 4/5: Loading local embedding function (all-MiniLM-L6-v2)...")
    embedding_function = get_embedding_function()
 
    print("Step 5/5: Writing to persistent ChromaDB store...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
 
    # Reset the collection if it already exists, so re-runs don't duplicate data.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
 
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )
 
    documents = [c["document"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [c["id"] for c in chunks]
 
    if not documents:
        print("No documents to add. Aborting.")
        return
 
    # Add in batches to keep memory usage reasonable for larger datasets.
    batch_size = 100
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"  -> Added batch {start}-{min(end, len(documents))}")
 
    print(f"Done. Persisted {len(documents)} recipe chunks to '{CHROMA_DB_PATH}' "
          f"in collection '{COLLECTION_NAME}'.")
 
 
if __name__ == "__main__":
    build_vector_store()