# src/data/ingest.py

import pandas as pd
import os

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw complaints CSV and keep only relevant columns.
    Drops rows where complaint narrative is missing.
    Returns a cleaned DataFrame with columns: text, label.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Keep only relevant columns
    df = df[['Consumer complaint narrative', 'Product']]

    # Drop missing narratives
    df = df.dropna(subset=['Consumer complaint narrative'])

    # Rename to snake_case
    df = df.rename(columns={
        'Consumer complaint narrative': 'text',
        'Product': 'label'
    })

    df = df.reset_index(drop=True)

    print(f"Loaded {len(df)} rows after dropping missing narratives.")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    return df


if __name__ == "__main__":
    df = load_raw_data('data/raw/complaints_raw.csv')
    df.to_csv('data/processed/complaints_clean.csv', index=False)
    print("Saved to data/processed/complaints_clean.csv")