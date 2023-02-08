from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version
from typing import Dict

app = FastAPI()


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    language: str


@app.get("/", tags=['home'])
def home() -> Dict[str, str]:
    return {"health_check": "OK", "model_version": model_version, "message": "Welcome to the language detection API"}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn) -> PredictionOut:
    language = predict_pipeline(payload.text)
    return PredictionOut(language=language)
