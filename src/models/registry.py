# src/models/registry.py

import os
import json
import shutil


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
        f1 = metrics.get('weighted_f1', 0)
        if f1 > best_f1:
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

    print(f"Champion: {best_name}")
    print(f"Weighted F1: {best_f1:.4f}")
    print(f"Model promoted to {champion_dir}/model.pkl")


if __name__ == "__main__":
    promote_champion()