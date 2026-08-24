# src/api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipelines.inference_pipeline import load_model

app = FastAPI(
    title="Customer Complaint Classifier",
    description="Classifies customer complaints into product categories.",
    version="1.0.0"
)


class ComplaintRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    text: str
    predicted_label: str
    confidence_score: float


@app.get("/")
def root():
    return {"message": "Customer Complaint Classifier API is running."}


@app.post("/predict", response_model=PredictionResponse)
def predict_complaint(request: ComplaintRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    model = load_model()
    prediction = model.predict([request.text])[0]
    confidence = max(model.decision_function([request.text])[0])
    return {
        'text': request.text,
        'predicted_label': prediction,
        'confidence_score': round(float(confidence), 3)
    }