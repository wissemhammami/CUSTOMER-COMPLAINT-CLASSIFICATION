# src/pipelines/evaluation_pipeline.py

from src.models.evaluate import evaluate_model
import os


def run_evaluation_pipeline(model_path: str = 'models/latest/model.pkl',
                            output_dir: str = 'models/latest') -> None:
    """
    Standalone evaluation pipeline.
    Loads champion model and evaluates on test set.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("=" * 50)
    print("EVALUATION PIPELINE")
    print("=" * 50)

    metrics = evaluate_model(model_path=model_path, output_dir=output_dir)

    print(f"Test weighted F1: {metrics['weighted_f1_test']}")
    print("EVALUATION COMPLETE")


if __name__ == "__main__":
    run_evaluation_pipeline()