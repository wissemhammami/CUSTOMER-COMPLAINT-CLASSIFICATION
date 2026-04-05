# src/pipelines/training_pipeline.py

from src.data.ingest import load_raw_data
from src.data.validate import validate_data
from src.features.build import build_features
from src.models.train import train_and_evaluate
from src.models.registry import promote_champion


def run_training_pipeline(raw_data_path: str = 'data/raw/complaints_raw.csv',
                          processed_dir: str = 'data/processed',
                          artifacts_dir: str = 'models_artifacts',
                          champion_dir:  str = 'models/latest') -> None:
    """
    Full training pipeline:
    ingest → validate → build features → train → promote champion
    """
    print("=" * 50)
    print("STEP 1 — Data ingestion")
    print("=" * 50)
    df = load_raw_data(raw_data_path)
    df.to_csv(f'{processed_dir}/complaints_clean.csv', index=False)

    print("\n" + "=" * 50)
    print("STEP 2 — Data validation")
    print("=" * 50)
    df = validate_data(df)
    df.to_csv(f'{processed_dir}/complaints_validated.csv', index=False)

    print("\n" + "=" * 50)
    print("STEP 3 — Feature engineering")
    print("=" * 50)
    build_features(
        input_path=f'{processed_dir}/complaints_validated.csv',
        output_dir=processed_dir
    )

    print("\n" + "=" * 50)
    print("STEP 4 — Model training")
    print("=" * 50)
    results = train_and_evaluate(output_dir=artifacts_dir)
    champion = max(results, key=results.get)
    print(f"Best model on eval set: {champion} — F1: {results[champion]:.4f}")

    print("\n" + "=" * 50)
    print("STEP 5 — Promote champion")
    print("=" * 50)
    promote_champion(artifacts_dir=artifacts_dir, champion_dir=champion_dir)

    print("\n" + "=" * 50)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    run_training_pipeline()