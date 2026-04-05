# src/data/validate.py

import pandas as pd


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate cleaned DataFrame.
    - Checks for missing values
    - Removes duplicates
    - Removes very short texts (less than 50 characters)
    - Resets index
    Returns validated DataFrame.
    """
    print(f"Shape before validation: {df.shape}")

    # Check missing values
    missing = df.isna().sum()
    print(f"Missing values:\n{missing}")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicates.")

    # Remove very short texts
    before = len(df)
    df = df[df['text'].str.len() >= 50]
    print(f"Removed {before - len(df)} short texts.")

    # Reset index
    df = df.reset_index(drop=True)

    print(f"Shape after validation: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    return df


if __name__ == "__main__":
    df = pd.read_csv('data/processed/complaints_clean.csv')
    df = validate_data(df)
    df.to_csv('data/processed/complaints_validated.csv', index=False)
    print("Saved to data/processed/complaints_validated.csv")