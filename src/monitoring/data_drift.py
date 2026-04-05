# src/monitoring/data_drift.py

import pandas as pd
import os


def run_drift_report(reference_path: str = 'data/processed/train_features.csv',
                     current_path:   str = 'data/processed/test_features.csv',
                     output_dir:     str = 'monitoring_reports') -> None:
    """
    Compare basic text statistics between training and test data.
    Saves a simple drift report to monitoring_reports/.
    """
    reference = pd.read_csv(reference_path).squeeze()
    current   = pd.read_csv(current_path).squeeze()

    ref_lengths = reference.str.len()
    cur_lengths = current.str.len()

    ref_word_counts = reference.str.split().apply(len)
    cur_word_counts = current.str.split().apply(len)

    report = pd.DataFrame({
        'metric': [
            'sample_count',
            'avg_text_length',
            'avg_word_count',
            'min_text_length',
            'max_text_length',
        ],
        'training': [
            len(reference),
            round(ref_lengths.mean(), 2),
            round(ref_word_counts.mean(), 2),
            ref_lengths.min(),
            ref_lengths.max(),
        ],
        'test': [
            len(current),
            round(cur_lengths.mean(), 2),
            round(cur_word_counts.mean(), 2),
            cur_lengths.min(),
            cur_lengths.max(),
        ]
    })

    report['difference'] = (report['test'] - report['training']).round(2)

    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/data_drift_report.csv'
    report.to_csv(output_path, index=False)

    print(report.to_string(index=False))
    print(f"\nDrift report saved to {output_path}")


if __name__ == "__main__":
    run_drift_report()