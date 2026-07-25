"""
03_chunking.py
 
Converts cleaned recipe dictionaries into unified text "chunks" that
will be embedded and stored in the vector database. Each chunk is a
single string document plus an attached metadata dictionary used for
citations at answer time.
 
No API keys are required or used in this module.
"""
 
 
def create_recipe_chunks(cleaned_recipes: list) -> list:
    """
    Format each cleaned recipe into a single text document block and
    attach citation metadata.
 
    Args:
        cleaned_recipes (list[dict]): Output of clean_recipe_data(), each
            item containing 'Title', 'Ingredients', 'Directions', 'Source'.
 
    Returns:
        list[dict]: A list of chunk dictionaries, each shaped as:
            {
                "id": str,
                "document": str,   # the unified text block
                "metadata": {"title": str, "source": str}
            }
    """
    chunks = []
 
    for idx, recipe in enumerate(cleaned_recipes):
        title = recipe.get("Title", "Untitled Recipe")
        ingredients = recipe.get("Ingredients", "")
        directions = recipe.get("Directions", "")
        source = recipe.get("Source", "Unknown Source")
 
        document_text = (
            f"Recipe Title: {title}\n"
            f"Ingredients: {ingredients}\n"
            f"Directions: {directions}"
        )
 
        metadata = {
            "title": title,
            "source": source,
        }
 
        chunks.append(
            {
                "id": f"recipe_{idx}",
                "document": document_text,
                "metadata": metadata,
            }
        )
 
    return chunks
 
 
if __name__ == "__main__":
    sample_cleaned = [
        {
            "Title": "Chocolate Cake",
            "Ingredients": "flour, sugar, cocoa powder, eggs",
            "Directions": "Mix ingredients. Bake at 350F for 30 minutes.",
            "Source": "https://example.com/chocolate-cake",
        }
    ]
    chunks = create_recipe_chunks(sample_cleaned)
    for c in chunks:
        print(c)