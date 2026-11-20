"""
Road Accident Severity Prediction API
Model : Tuned Random Forest (imblearn Pipeline) + threshold rules for class 1 and class 4
Run   : python -m uvicorn main:app --reload
Docs  : http://127.0.0.1:8000/docs
"""
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# ---- Load everything once, when the server starts ----
model = joblib.load(MODEL_DIR / "random_forest_tuned.pkl")
thresholds = joblib.load(MODEL_DIR / "thresholds.pkl")          # {"class1": x, "class4": y}
feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")  # list of column names used in training

SEVERITY_LABELS = {1: "Minor", 2: "Moderate", 3: "Serious", 4: "Most severe"}

app = FastAPI(
    title="Road Accident Severity Prediction API",
    description="Predicts accident severity (1-4) using a tuned Random Forest with threshold tuning.",
    version="1.0",
)

# Lets a browser-based HTML page call this API later (needed for the project's HTML front end)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AccidentInput(BaseModel):
    # column name -> numeric value, exactly as the model saw it in X_train.csv
    features: Dict[str, float]


@app.get("/")
def root():
    return {"message": "Road Accident Severity API is running. Open /docs to test it."}


@app.get("/health")
def health():
    return {"status": "ok", "n_features": len(feature_columns), "thresholds": thresholds}


@app.get("/features")
def features():
    """Lists the exact column names the model expects."""
    return {"feature_columns": feature_columns}


@app.post("/predict")
def predict(data: AccidentInput):
    unknown = [c for c in data.features if c not in feature_columns]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown feature(s): {unknown}. See GET /features.")

    # Build one row in the same column order as training; columns not sent are filled with 0
    row = pd.DataFrame([data.features]).reindex(columns=feature_columns, fill_value=0)

    proba = model.predict_proba(row)[0]
    classes = [int(c) for c in model.classes_]

    # Same logic as notebook section 6
    prediction = classes[int(np.argmax(proba))]
    if proba[0] >= thresholds["class1"]:
        prediction = 1
    if proba[3] >= thresholds["class4"]:
        prediction = 4

    return {
        "predicted_severity": int(prediction),
        "severity_label": SEVERITY_LABELS.get(int(prediction), "Unknown"),
        "probabilities": {f"class_{c}": round(float(p), 4) for c, p in zip(classes, proba)},
    }
