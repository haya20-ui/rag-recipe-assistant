"""
04_vector_representation.py
 
Defines the embedding function used to turn recipe text chunks into
vector embeddings. Uses a local Sentence-Transformers model
(all-MiniLM-L6-v2) via ChromaDB's built-in embedding function wrapper,
so no external API key or network call is required at query/embedding
time (the model weights are downloaded once from Hugging Face on first
run and cached locally).
 
No API keys are required or used in this module.
"""
 
import chromadb.utils.embedding_functions as embedding_functions
 
 
def get_embedding_function():
    """
    Build and return a local Sentence-Transformers embedding function
    for use with ChromaDB collections.
 
    Returns:
        chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction
    """
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return embedding_function
 
 
if __name__ == "__main__":
    ef = get_embedding_function()
    sample_vectors = ef(["Recipe Title: Chocolate Cake\nIngredients: flour, sugar"])
    print(f"Generated {len(sample_vectors)} embedding(s).")
    print(f"Embedding dimension: {len(sample_vectors[0])}")