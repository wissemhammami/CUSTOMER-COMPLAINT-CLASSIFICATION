# src/models/train.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from src.features.transformers import TextCleaner
import joblib
import json
import os
from datetime import datetime
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]


def load_configs() -> tuple:
    with open(ROOT_DIR / 'configs' / 'model.yaml') as f:
        model_config = yaml.safe_load(f)
    with open(ROOT_DIR / 'configs' / 'training.yaml') as f:
        training_config = yaml.safe_load(f)
    return model_config, training_config


def build_models(model_config: dict) -> dict:
    return {
        'logistic_regression': LogisticRegression(
            **model_config['models']['logistic_regression']['init_params']
        ),
        'naive_bayes': MultinomialNB(
            **model_config['models']['naive_bayes']['init_params']
        ),
        'linear_svm': LinearSVC(
            **model_config['models']['linear_svm']['init_params']
        ),
    }


def build_param_grids(model_config: dict) -> dict:
    return {
        name: {f'clf__{parameter}': values for parameter, values in config['param_grid'].items()}
        for name, config in model_config['models'].items()
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

    model_config, training_config = load_configs()
    models = build_models(model_config)
    param_grids = build_param_grids(model_config)
    cv = StratifiedKFold(**training_config['cv'])
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")

        pipeline = Pipeline([
            ('cleaner', TextCleaner()),
            ('tfidf', TfidfVectorizer(
                max_features=model_config['tfidf']['max_features'],
                ngram_range=tuple(model_config['tfidf']['ngram_range']),
            )),
            ('clf', model),
        ], memory=os.path.join(output_dir, 'pipeline_cache'))

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            scoring=training_config['scoring'],
            cv=cv,
            n_jobs=2,
            refit=True,
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        y_pred = pipeline.predict(X_eval)

        f1     = f1_score(y_eval, y_pred, average='weighted')
        report = classification_report(y_eval, y_pred)

        print(f"{name} weighted F1: {f1:.4f}")
        print(f"Best parameters: {search.best_params_}")

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