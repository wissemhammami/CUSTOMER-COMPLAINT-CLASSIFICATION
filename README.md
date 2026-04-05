```markdown
# Customer Complaint Classifier

Automatically classifies consumer complaints into financial product categories using Machine Learning.

## Live Demo
[Coming soon]

## Dataset
- **Source:** CFPB Consumer Complaints Dataset
- **Size:** 245,228 complaints after cleaning
- **Input:** Free-text consumer complaint narrative
- **Output:** Product category (5 classes)

| Category | Count |
|---|---|
| Checking or savings account | 100,420 |
| Money transfer / virtual currency | 58,335 |
| Mortgage | 36,566 |
| Vehicle loan or lease | 25,345 |
| Student loan | 24,562 |

## Results

| Model | Weighted F1 |
|---|---|
| **Logistic Regression** | **0.8983** |
| Linear SVM | 0.8919 |
| Naive Bayes | 0.8313 |

Champion: **Logistic Regression** — Test F1: **0.8971**

## Project Structure

```
├── configs/            # Model and training configuration
├── data/               # Raw and processed data (gitignored)
├── models/latest/      # Champion model
├── models_artifacts/   # Experiment artifacts (gitignored)
├── monitoring_reports/ # Data drift reports (gitignored)
├── notebooks/          # Step-by-step experiment notebooks
├── src/
│   ├── data/           # Ingestion and validation
│   ├── features/       # Text cleaning and TF-IDF
│   ├── models/         # Training, evaluation, registry
│   ├── monitoring/     # Data drift detection
│   ├── pipelines/      # End-to-end pipeline runners
│   └── api/            # FastAPI REST API
└── app.py              # Streamlit demo
```

## Setup

```bash
git clone https://github.com/wissemhammami/CUSTOMER-COMPLAINT-CLASSIFICATION
cd CUSTOMER-COMPLAINT-CLASSIFICATION
python -m venv env3
env3\Scripts\activate
pip install -r requirements.txt
```

## Run

**Full training pipeline:**
```bash
python -m src.pipelines.training_pipeline
```

**Evaluation:**
```bash
python -m src.pipelines.evaluation_pipeline
```

**API:**
```bash
uvicorn src.api.main:app --reload
```
API docs: `http://127.0.0.1:8000/docs`

**Streamlit app:**
```bash
streamlit run app.py
```

**Data drift report:**
```bash
python -m src.monitoring.data_drift
```

## Tech Stack
Python, scikit-learn, NLTK, FastAPI, Streamlit, Evidently, pandas, numpy
```

