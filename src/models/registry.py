# src/models/registry.py

import os
import json
import shutil
from datetime import datetime, timezone

import joblib


def get_best_artifact(artifacts_dir: str = 'models_artifacts') -> tuple:
    """
    Scan all artifact folders, compare weighted_f1 from metrics.json,
    return the best model name and its artifact path.
    """
    best_f1   = 0
    best_path = None
    best_name = None

    for folder in os.listdir(artifacts_dir):
        metrics_path = os.path.join(artifacts_dir, folder, 'metrics.json')
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        f1 = round(metrics.get('weighted_f1', 0), 4)
        if f1 > best_f1 or (f1 == best_f1 and (best_name is None or folder > best_name)):
            best_f1   = f1
            best_path = os.path.join(artifacts_dir, folder)
            best_name = folder

    return best_name, best_path, best_f1


def promote_champion(artifacts_dir: str = 'models_artifacts',
                     champion_dir: str = 'models/latest') -> None:
    """
    Find the best model artifact and copy it to models/latest/.
    """
    best_name, best_path, best_f1 = get_best_artifact(artifacts_dir)

    if best_path is None:
        raise FileNotFoundError("No valid artifacts found.")

    os.makedirs(champion_dir, exist_ok=True)
    shutil.copy(os.path.join(best_path, 'model.pkl'),
                os.path.join(champion_dir, 'model.pkl'))

    write_model_metadata(
        model_path=os.path.join(champion_dir, 'model.pkl'),
        champion_name=best_name,
        weighted_f1=best_f1,
        champion_dir=champion_dir,
    )

    print(f"Champion: {best_name}")
    print(f"Weighted F1: {best_f1:.4f}")
    print(f"Model promoted to {champion_dir}/model.pkl")


def write_model_metadata(model_path: str, champion_name: str,
                         weighted_f1: float, champion_dir: str) -> None:
    """Write metadata alongside the promoted model artifact."""
    model = joblib.load(model_path)
    classifier = model.named_steps['clf']
    vectorizer = model.named_steps['tfidf']
    model_type = {
        'LogisticRegression': 'Logistic Regression',
        'LinearSVC': 'Linear SVM',
        'MultinomialNB': 'Multinomial Naive Bayes',
    }.get(type(classifier).__name__, type(classifier).__name__)
    metadata = {
        'model_name': champion_name,
        'model_type': model_type,
        'promotion_timestamp': datetime.now(timezone.utc).isoformat(),
        'training_date': datetime.now(timezone.utc).date().isoformat(),
        'dataset': 'CFPB complaints',
        'weighted_f1': round(weighted_f1, 4),
        'pipeline_steps': list(model.named_steps),
        'hyperparameters': {
            'tfidf_max_features': vectorizer.max_features,
            'tfidf_ngram_range': list(vectorizer.ngram_range),
            **classifier.get_params(),
        },
    }
    with open(os.path.join(champion_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    promote_champion()