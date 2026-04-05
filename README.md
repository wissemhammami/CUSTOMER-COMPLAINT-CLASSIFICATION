# Customer Complaint Classifier

End-to-end NLP machine learning project that automatically classifies consumer financial complaints into product categories.
Built with Logistic Regression, TF-IDF, FastAPI, and Streamlit.

## Live Demo

[Open the app](https://complaint-classifier-wissem.streamlit.app)

---

## Problem Statement

Financial regulators and companies receive thousands of consumer complaints daily.
Manually routing each complaint to the right team is slow and error-prone.
This project automatically classifies free-text complaints into product categories
so they can be routed and prioritized instantly.

**Dataset** : [CFPB Consumer Complaints — public dataset](https://www.consumerfinance.gov/data-research/consumer-complaints/)  
**Target** : Multi-class classification — 5 financial product categories  
**Dataset size** : 245,228 complaints after cleaning

---

## Results

| Model | Weighted F1 (Eval) | Weighted F1 (Test) |
|---|---|---|
| **Logistic Regression** | **0.8983** | **0.8971** |
| Linear SVM | 0.8919 | — |
| Multinomial Naive Bayes | 0.8313 | — |

### Per-class performance (Test Set)

| Category | Precision | Recall | F1 |
|---|---|---|---|
| Checking or savings account | 0.86 | 0.93 | 0.89 |
| Money transfer / virtual currency | 0.88 | 0.78 | 0.83 |
| Mortgage | 0.96 | 0.95 | 0.95 |
| Student loan | 0.98 | 0.95 | 0.96 |
| Vehicle loan or lease | 0.93 | 0.92 | 0.92 |

---

## Project Structure
```
CUSTOMER_COMPLAINT_CLASSIFICATION/
│
├── configs/
│   ├── model.yaml                   # Model hyperparameters
│   └── training.yaml                # Training configuration
│
├── data/
│   ├── raw/                         # Raw CSV (gitignored)
│   └── processed/                   # Cleaned splits (gitignored)
│
├── models/
│   └── latest/                      # Champion model served by API and Streamlit
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
│   │   ├── ingest.py                # Load and clean raw data
│   │   └── validate.py              # Remove duplicates and short texts
│   ├── features/
│   │   ├── transformers.py          # Custom sklearn TextCleaner transformer
│   │   └── build.py                 # Train/eval/test split and feature saving
│   ├── models/
│   │   ├── train.py                 # Train NB, LR, SVM — save artifacts
│   │   ├── evaluate.py              # Evaluate on test set — save metrics
│   │   └── registry.py              # Select and promote champion model
│   ├── monitoring/
│   │   └── data_drift.py            # Compare train vs test text statistics
│   ├── pipelines/
│   │   ├── training_pipeline.py     # End-to-end training runner
│   │   ├── evaluation_pipeline.py   # Standalone evaluation runner
│   │   └── inference_pipeline.py    # Load model and predict
│   └── api/
│       └── main.py                  # FastAPI REST API
│
├── app.py                           # Streamlit web application
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation
```bash
git clone https://github.com/wissemhammami/CUSTOMER-COMPLAINT-CLASSIFICATION
cd CUSTOMER-COMPLAINT-CLASSIFICATION

python -m venv env
env\Scripts\activate        # Windows
source env/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

Place the raw dataset at:
```
data/raw/complaints_raw.csv
```

---

## How to Run

**1 — Full training pipeline**
```bash
python -m src.pipelines.training_pipeline
```
Runs: ingest → validate → build features → train 3 models → promote champion

**2 — Evaluation**
```bash
python -m src.pipelines.evaluation_pipeline
```

**3 — Launch the API**
```bash
uvicorn src.api.main:app --reload
```
API docs available at `http://127.0.0.1:8000/docs`

**4 — Launch the Streamlit app**
```bash
streamlit run app.py
```
App available at `http://localhost:8501`

**5 — Data drift report**
```bash
python -m src.monitoring.data_drift
```
Saves report to `monitoring_reports/data_drift_report.csv`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Check API status |
| POST | `/predict` | Classify a single complaint |

**Example request**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice for the same transaction and my bank refuses to refund me."}'
```

**Example response**
```json
{
  "text": "I was charged twice for the same transaction and my bank refuses to refund me.",
  "predicted_label": "Checking or savings account",
  "confidence_score": 5.599
}
```

---

## ML Pipeline
```
Raw Text
  └── Ingestion              → Load CSV, keep text + label columns
        └── Validation       → Remove duplicates, drop short texts
              └── Cleaning   → Lowercase, remove punctuation, remove stopwords
                    └── TF-IDF (50k features, unigrams + bigrams)
                          └── Model comparison → NB / LR / LinearSVC
                                └── Champion selection → Logistic Regression (F1: 0.8971)
```

---

## Key Insights

- Logistic Regression outperformed LinearSVC — contrary to the common assumption that SVM wins on text classification
- Money transfer is the hardest category to classify (F1: 0.83) due to overlap with banking complaints
- Mortgage and Student loan are the easiest (F1: 0.95–0.96) — highly distinct vocabulary
- Removing duplicates (28,720 rows) was critical — keeping them would have inflated metrics artificially

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML | scikit-learn, NLTK |
| API | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit |
| Monitoring | Evidently |
| Visualization | Matplotlib, Seaborn |
| Serialization | joblib |

---

## Author

**Wissem Hammami**  
Engineering Student — ESSAI, University of Carthage, Tunisia  
[GitHub](https://github.com/wissemhammami)