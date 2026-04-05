````markdown
# Customer Complaint Classifier

Automatically classifies consumer financial complaints into product categories using a Machine Learning pipeline built on the CFPB dataset.

> **Live Demo:** [complaint-classifier-wissem.streamlit.app](https://complaint-classifier-wissem.streamlit.app)

---

## Overview

This project demonstrates an end-to-end NLP classification pipeline — from raw text ingestion to a deployed web application. Given a free-text consumer complaint, the system predicts which financial product category it belongs to.

- **Input:** Raw consumer complaint narrative
- **Output:** Product category + confidence score
- **Dataset:** CFPB Consumer Complaints (public dataset)
- **Deployment:** Streamlit Cloud + FastAPI REST API

---

## Results

| Model | Weighted F1 (Eval) | Weighted F1 (Test) |
|---|---|---|
| **Logistic Regression** | **0.8983** | **0.8971** |
| Linear SVM | 0.8919 | — |
| Multinomial Naive Bayes | 0.8313 | — |

Champion model: **Logistic Regression** with TF-IDF (unigrams + bigrams, 50,000 features)

### Per-class performance (Test Set)

| Category | Precision | Recall | F1 |
|---|---|---|---|
| Checking or savings account | 0.86 | 0.93 | 0.89 |
| Money transfer / virtual currency | 0.88 | 0.78 | 0.83 |
| Mortgage | 0.96 | 0.95 | 0.95 |
| Student loan | 0.98 | 0.95 | 0.96 |
| Vehicle loan or lease | 0.93 | 0.92 | 0.92 |

---

## Dataset

| Property | Value |
|---|---|
| Source | CFPB Consumer Complaints |
| Raw rows | 476,303 |
| After cleaning | 245,228 |
| Features | Free-text complaint narrative |
| Target | Product category (5 classes) |
| Split | 70% train / 15% eval / 15% test |

| Category | Count |
|---|---|
| Checking or savings account | 100,420 |
| Money transfer / virtual currency | 58,335 |
| Mortgage | 36,566 |
| Vehicle loan or lease | 25,345 |
| Student loan | 24,562 |

---

## Project Structure

```
CUSTOMER_COMPLAINT_CLASSIFICATION/
│
├── configs/
│   ├── model.yaml               # Model hyperparameters
│   └── training.yaml            # Training configuration
│
├── data/
│   ├── raw/                     # Raw CSV (gitignored)
│   └── processed/               # Cleaned splits (gitignored)
│
├── models/
│   └── latest/                  # Champion model served by API and Streamlit
│       ├── model.pkl
│       ├── metrics.json
│       ├── classification_report.txt
│       ├── confusion_matrix.png
│       └── eval_predictions.csv
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_text_preprocessing.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_evaluation_experiments.ipynb
│   └── 06_inference_experiments.ipynb
│
├── src/
│   ├── data/
│   │   ├── ingest.py            # Load and clean raw data
│   │   └── validate.py          # Remove duplicates and short texts
│   ├── features/
│   │   ├── transformers.py      # Custom sklearn TextCleaner transformer
│   │   └── build.py             # Train/eval/test split and feature saving
│   ├── models/
│   │   ├── train.py             # Train NB, LR, SVM — save artifacts
│   │   ├── evaluate.py          # Evaluate on test set — save metrics
│   │   └── registry.py          # Select and promote champion model
│   ├── monitoring/
│   │   └── data_drift.py        # Compare train vs test text statistics
│   ├── pipelines/
│   │   ├── training_pipeline.py    # End-to-end training runner
│   │   ├── evaluation_pipeline.py  # Standalone evaluation runner
│   │   └── inference_pipeline.py   # Load model and predict
│   └── api/
│       └── main.py              # FastAPI REST API
│
├── app.py                       # Streamlit web application
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup

```bash
git clone https://github.com/wissemhammami/CUSTOMER-COMPLAINT-CLASSIFICATION
cd CUSTOMER-COMPLAINT-CLASSIFICATION
python -m venv env3
env3\Scripts\activate        # Windows
source env3/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

Place the raw dataset at:
```
data/raw/complaints_raw.csv
```

---

## Run

### Full training pipeline
```bash
python -m src.pipelines.training_pipeline
```
Runs: ingest → validate → build features → train 3 models → promote champion

### Evaluation
```bash
python -m src.pipelines.evaluation_pipeline
```

### FastAPI
```bash
uvicorn src.api.main:app --reload
```
Swagger docs available at `http://127.0.0.1:8000/docs`

**Example request:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice for the same transaction and my bank refuses to refund me."}'
```

**Example response:**
```json
{
  "text": "I was charged twice for the same transaction and my bank refuses to refund me.",
  "predicted_label": "Checking or savings account",
  "confidence_score": 5.599
}
```

### Streamlit app
```bash
streamlit run app.py
```

### Data drift report
```bash
python -m src.monitoring.data_drift
```
Saves report to `monitoring_reports/data_drift_report.csv`

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| scikit-learn | TF-IDF, model training, evaluation |
| NLTK | Text preprocessing, stopwords |
| FastAPI | REST API |
| Streamlit | Web application |
| Evidently | Data drift monitoring |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualization |
| joblib | Model serialization |

---

## Author

**Wissem Hammami**  
Engineering Student — ESSAI, University of Carthage, Tunisia  
[GitHub](https://github.com/wissemhammami)
````

