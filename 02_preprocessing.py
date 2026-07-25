"""
02_preprocessing.py
 
Cleans and normalizes the raw recipe dictionaries produced by
01_documents.py. Handles whitespace stripping, sanitizing text fields,
and normalizing column names so downstream modules can rely on a
consistent schema: Title, Ingredients, Directions, Source.
 
No API keys are required or used in this module.
"""
 
import re
 
 
# Candidate column names (case-insensitive) that might appear in the
# source CSV for each canonical field. This makes the cleaner resilient
# to slightly different Recipe dataset schemas.
FIELD_ALIASES = {
    "Title": ["title", "recipe_title", "name", "recipe_name"],
    "Ingredients": ["ingredients", "ingredient_list", "recipe_ingredients"],
    "Directions": ["directions", "instructions", "steps", "recipe_instructions"],
    "Source": ["source", "url", "link", "recipe_url", "site"],
}
 
 
def _find_field(recipe: dict, canonical_name: str) -> str:
    """Find a field value in `recipe` using known aliases (case-insensitive)."""
    lower_map = {k.lower(): k for k in recipe.keys()}
 
    for alias in FIELD_ALIASES.get(canonical_name, []):
        if alias.lower() in lower_map:
            return recipe.get(lower_map[alias.lower()], "")
 
    # Fall back to a direct case-sensitive match on the canonical name.
    return recipe.get(canonical_name, "")
 
 
def _sanitize_text(value) -> str:
    """Strip whitespace, collapse repeated whitespace, and coerce to string."""
    if value is None:
        return ""
    text = str(value)
    text = text.strip()
    # Collapse multiple whitespace/newlines into a single space for cleanliness,
    # while still keeping the text readable.
    text = re.sub(r"\s+", " ", text)
    return text
 
 
def clean_recipe_data(recipes_list: list) -> list:
    """
    Clean and normalize a list of raw recipe dictionaries.
 
    Args:
        recipes_list (list[dict]): Raw recipe dictionaries from load_documents().
 
    Returns:
        list[dict]: Cleaned recipe dictionaries with normalized keys:
            'Title', 'Ingredients', 'Directions', 'Source'.
    """
    cleaned_recipes = []
 
    for recipe in recipes_list:
        title = _sanitize_text(_find_field(recipe, "Title")) or "Untitled Recipe"
        ingredients = _sanitize_text(_find_field(recipe, "Ingredients"))
        directions = _sanitize_text(_find_field(recipe, "Directions"))
        source = _sanitize_text(_find_field(recipe, "Source")) or "Unknown Source"
 
        # Skip completely empty rows (no title, no ingredients, no directions).
        if not ingredients and not directions and title == "Untitled Recipe":
            continue
 
        cleaned_recipes.append(
            {
                "Title": title,
                "Ingredients": ingredients,
                "Directions": directions,
                "Source": source,
            }
        )
 
    return cleaned_recipes
 
 
if __name__ == "__main__":
    # Small smoke test with dummy data.
    sample_raw = [
        {
            "title": "  Chocolate Cake  ",
            "ingredients": "flour, sugar,   cocoa powder \n eggs",
            "directions": "Mix.   Bake at 350F.",
            "url": "https://example.com/chocolate-cake",
        }
    ]
    result = clean_recipe_data(sample_raw)
    print(result)