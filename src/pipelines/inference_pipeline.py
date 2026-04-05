# src/pipelines/inference_pipeline.py

import joblib
import numpy as np
import os


def load_model(model_path: str = 'models/latest/model.pkl'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def predict(texts: list, model_path: str = 'models/latest/model.pkl') -> list:
    """
    Accept a list of raw complaint texts.
    Returns a list of dicts with predicted_label and confidence_score.
    """
    model       = load_model(model_path)
    predictions = model.predict(texts)
    confidence  = np.max(model.decision_function(texts), axis=1)

    results = []
    for text, label, score in zip(texts, predictions, confidence):
        results.append({
            'text':            text,
            'predicted_label': label,
            'confidence_score': round(float(score), 3)
        })

    return results


if __name__ == "__main__":
    sample = [
        "I have been charged twice for the same transaction on my checking account.",
        "My mortgage payment was incorrectly applied to the wrong account.",
        "I sent money through a wire transfer and it never arrived.",
        "My student loan servicer keeps changing and I lost track of my balance.",
        "The dealership added extra fees to my auto loan without my consent."
    ]

    results = predict(sample)
    for r in results:
        print(f"Text      : {r['text'][:60]}...")
        print(f"Prediction: {r['predicted_label']}")
        print(f"Confidence: {r['confidence_score']}")
        print()