# src/models/train.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report
import joblib
import json
import os
from datetime import datetime


def get_models() -> dict:
    return {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'naive_bayes': MultinomialNB(),
        'linear_svm': LinearSVC(random_state=42, max_iter=2000)
    }


def train_and_evaluate(output_dir: str = 'models_artifacts') -> dict:
    """
    Train all models on train split, evaluate on eval split.
    Save each model artifact with metrics and classification report.
    Returns dict of model_name -> weighted_f1.
    """
    X_train = pd.read_csv('data/processed/train_features.csv').squeeze()
    X_eval  = pd.read_csv('data/processed/eval_features.csv').squeeze()
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_eval  = pd.read_csv('data/processed/y_eval.csv').squeeze()

    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
            ('clf', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_eval)

        f1     = f1_score(y_eval, y_pred, average='weighted')
        report = classification_report(y_eval, y_pred)

        print(f"{name} weighted F1: {f1:.4f}")

        # Save artifact
        timestamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
        artifact_dir = f'{output_dir}/{name}_champion_{timestamp}'
        os.makedirs(artifact_dir, exist_ok=True)

        joblib.dump(pipeline, f'{artifact_dir}/model.pkl')

        with open(f'{artifact_dir}/metrics.json', 'w') as f:
            json.dump({'model': name, 'weighted_f1': round(f1, 4)}, f, indent=2)

        with open(f'{artifact_dir}/classification_report.txt', 'w') as f:
            f.write(report)

        results[name] = f1
        print(f"Artifact saved: {artifact_dir}\n")

    return results


if __name__ == "__main__":
    results = train_and_evaluate()
    champion = max(results, key=results.get)
    print(f"Champion: {champion} — F1: {results[champion]:.4f}")