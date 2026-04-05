# src/models/evaluate.py

import pandas as pd
import joblib
import json
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os


def evaluate_model(model_path: str, output_dir: str = None) -> dict:
    """
    Load a saved model pipeline and evaluate it on the test set.
    Saves metrics, classification report, confusion matrix and predictions.
    Returns dict with evaluation metrics.
    """
    X_test = pd.read_csv('data/processed/test_features.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

    model  = joblib.load(model_path)
    y_pred = model.predict(X_test)

    f1     = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)

    print(f"Test weighted F1: {f1:.4f}")
    print(report)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        with open(f'{output_dir}/metrics.json', 'w') as f:
            json.dump({'weighted_f1_test': round(f1, 4)}, f, indent=2)

        with open(f'{output_dir}/classification_report.txt', 'w') as f:
            f.write(report)

        pd.DataFrame({
            'true_label':      y_test.values,
            'predicted_label': y_pred
        }).to_csv(f'{output_dir}/eval_predictions.csv', index=False)

        cm   = confusion_matrix(y_test, y_pred, labels=model.classes_)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
        plt.title('Confusion Matrix - Test Set')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=150)
        plt.close()
        print(f"Results saved to {output_dir}")

    return {'weighted_f1_test': round(f1, 4)}


if __name__ == "__main__":
    evaluate_model(
        model_path='models/latest/model.pkl',
        output_dir='models/latest'
    )