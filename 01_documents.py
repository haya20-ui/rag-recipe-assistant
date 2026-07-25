"""
01_documents.py
 
Responsible for loading the raw recipe data from the workspace CSV file
and converting it into a list of plain Python dictionaries that the rest
of the RAG pipeline can consume.
 
No API keys are required or used in this module.
"""
 
import os

import pandas as pd


def load_documents(csv_path: str = "1_Recipe_csv.csv", sample_size: int = 300):
    """
    Load recipe documents from a CSV file.
 
    Args:
        csv_path (str): Path to the recipe CSV file. Defaults to the
            workspace file '1_Recipe_csv.csv' located in the project root.
        sample_size (int): Maximum number of rows to sample from the CSV.
            If the CSV has fewer rows than sample_size, all rows are used.
 
    Returns:
        list[dict]: A list of recipe records as dictionaries, with all
            NaN values replaced by empty strings.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find CSV file at '{csv_path}'. "
            f"Please make sure '1_Recipe_csv.csv' is in the project root directory."
        )
 
    df = pd.read_csv(csv_path)
 
    # Fill any missing/NaN values with empty strings so downstream
    # string operations never fail on NaN/float types.
    df = df.fillna("")
 
    # Safely sample up to `sample_size` rows. If the dataframe has fewer
    # rows than requested, just use everything (no error).
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
 
    # Convert every row into a plain dictionary.
    recipes_list = df.to_dict(orient="records")
 
    return recipes_list
 
 
if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} recipe documents.")
    if docs:
        print("Sample document keys:", list(docs[0].keys()))
        print("Sample document:", docs[0])
 