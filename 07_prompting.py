"""
07_prompting.py
 
Builds the RAG prompt from retrieved recipe context and calls the
OpenRouter chat completions API to generate the final answer.
 
API key handling:
    - OPENROUTER_API_KEY is read from the environment variable
      'OPENROUTER_API_KEY' (e.g. via a local .env file loaded with
      python-dotenv). NOTHING is hardcoded in this file.
    - streamlit_app.py additionally falls back to st.secrets if the
      environment variable is not set (see that file for details).
"""
 
import os
import requests
from dotenv import load_dotenv
 
# Load variables from a local .env file if present (no-op if it doesn't exist).
load_dotenv()
 
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
 
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
 
 
def build_prompt(user_query: str, context_chunks: list) -> str:
    """
    Build a grounded RAG prompt instructing the LLM to answer using only
    the retrieved recipe context, and to cite recipe names/sources.
 
    Args:
        user_query (str): The user's original question.
        context_chunks (list[dict]): Output of query_recipes(), each item
            containing 'document' and 'metadata' (with 'title'/'source').
 
    Returns:
        str: The fully assembled prompt text.
    """
    if not context_chunks:
        context_text = "No relevant recipes were found in the database."
    else:
        formatted_chunks = []
        for i, chunk in enumerate(context_chunks, start=1):
            title = chunk.get("metadata", {}).get("title", "Unknown Recipe")
            source = chunk.get("metadata", {}).get("source", "Unknown Source")
            document = chunk.get("document", "")
            formatted_chunks.append(
                f"[Source {i}: \"{title}\" ({source})]\n{document}"
            )
        context_text = "\n\n---\n\n".join(formatted_chunks)
 
    prompt = f"""You are a helpful cooking assistant. Answer the user's question using ONLY the recipe context provided below. Do not make up information that is not present in the context.
 
When you use information from a recipe, you MUST cite it by name in your answer, for example: "According to the 'Chocolate Cake' recipe, ...". If multiple recipes are relevant, cite each one you use. If the context does not contain enough information to answer the question, say so honestly instead of guessing.
 
Context:
{context_text}
 
User Question: {user_query}
 
Answer (with citations to recipe names/sources):"""
 
    return prompt
 
 
def generate_response(prompt: str) -> str:
    """
    Call the OpenRouter chat completions API to generate a response.
 
    Args:
        prompt (str): The full prompt text (typically from build_prompt()).
 
    Returns:
        str: The generated answer text, or an error message if the call fails.
    """
    if not OPENROUTER_API_KEY:
        return (
            "Error: No OpenRouter API key was found. Please set the "
            "OPENROUTER_API_KEY environment variable (e.g. in a .env file) "
            "or, if running via Streamlit, in your app's secrets."
        )
 
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
 
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }
 
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        return answer
    except requests.exceptions.RequestException as e:
        return f"Error calling OpenRouter API: {e}"
    except (KeyError, IndexError) as e:
        return f"Error parsing OpenRouter API response: {e}"
 
 
if __name__ == "__main__":
    sample_context = [
        {
            "document": "Recipe Title: Chocolate Cake\nIngredients: flour, sugar, cocoa powder\nDirections: Mix and bake at 350F.",
            "metadata": {"title": "Chocolate Cake", "source": "https://example.com/chocolate-cake"},
        }
    ]
    test_prompt = build_prompt("How do I make a chocolate cake?", sample_context)
    print("=== PROMPT ===")
    print(test_prompt)
    print("\n=== RESPONSE ===")
    print(generate_response(test_prompt))