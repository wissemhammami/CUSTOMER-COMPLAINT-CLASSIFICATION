# src/features/build.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os


def build_features(input_path: str, output_dir: str) -> None:
    """
    Load validated data, split into train/eval/test,
    split raw text and save it to the output directory.
    Cleaning and TF-IDF are handled inside the persisted model pipeline.
    """
    df = pd.read_csv(input_path)

    X = df['text']
    y = df['label']

    # Train / eval / test split (70 / 15 / 15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_eval, X_test, y_eval, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)} | Eval: {len(X_eval)} | Test: {len(X_test)}")

    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(f'{output_dir}/train_features.csv', index=False)
    X_eval.to_csv(f'{output_dir}/eval_features.csv',   index=False)
    X_test.to_csv(f'{output_dir}/test_features.csv',   index=False)

    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_eval.to_csv(f'{output_dir}/y_eval.csv',   index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv',   index=False)

    print("All splits saved successfully.")


if __name__ == "__main__":
    build_features(
        input_path='data/processed/complaints_validated.csv',
        output_dir='data/processed'
    )