# src/api/main.py

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from src.pipelines.inference_pipeline import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield
    app.state.model = None


app = FastAPI(
    title="Customer Complaint Classifier",
    description="Classifies customer complaints into product categories.",
    version="1.0.0",
    lifespan=lifespan,
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
def predict_complaint(request: Request, complaint: ComplaintRequest):
    if not complaint.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    model = request.app.state.model
    prediction = model.predict([complaint.text])[0]
    confidence = max(model.decision_function([complaint.text])[0])
    return {
        'text': complaint.text,
        'predicted_label': prediction,
        'confidence_score': round(float(confidence), 3)
    }