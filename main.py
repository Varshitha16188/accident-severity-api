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
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# ---- Load everything once, when the server starts ----
model = joblib.load(MODEL_DIR / "random_forest_tuned.pkl")
thresholds = joblib.load(MODEL_DIR / "thresholds.pkl")          # {"class1": x, "class4": y}
feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")  # list of column names used in training

# ---- Example input (pre-filled so the faculty can run it directly) ----
EXAMPLE_FEATURES = {
    "num__Start_Lat": 38.58461,
    "num__Start_Lng": -121.503044,
    "num__Distance(mi)": 0.0,
    "num__Temperature(F)": 72.0,
    "num__Humidity(%)": 53.0,
    "num__Pressure(in)": 29.84,
    "num__Visibility(mi)": 10.0,
    "num__Wind_Speed(mph)": 13.8,
    "num__Precipitation(in)": 0.0,
    "num__Amenity": 0.0,
    "num__Bump": 0.0,
    "num__Crossing": 0.0,
    "num__Give_Way": 0.0,
    "num__Junction": 0.0,
    "num__No_Exit": 0.0,
    "num__Railway": 0.0,
    "num__Roundabout": 0.0,
    "num__Station": 0.0,
    "num__Stop": 0.0,
    "num__Traffic_Calming": 0.0,
    "num__Traffic_Signal": 0.0,
    "num__Hour": 16.0,
    "num__Day_of_Week": 3.0,
    "num__Month": 4.0,
    "num__Is_Weekend": 0.0,
    "num__Is_Day": 1.0,
    "cat__State_AL": 0.0,
    "cat__State_AR": 0.0,
    "cat__State_AZ": 0.0,
    "cat__State_CA": 1.0,
    "cat__State_CO": 0.0,
    "cat__State_CT": 0.0,
    "cat__State_DC": 0.0,
    "cat__State_DE": 0.0,
    "cat__State_FL": 0.0,
    "cat__State_GA": 0.0,
    "cat__State_IA": 0.0,
    "cat__State_ID": 0.0,
    "cat__State_IL": 0.0,
    "cat__State_IN": 0.0,
    "cat__State_KS": 0.0,
    "cat__State_KY": 0.0,
    "cat__State_LA": 0.0,
    "cat__State_MA": 0.0,
    "cat__State_MD": 0.0,
    "cat__State_ME": 0.0,
    "cat__State_MI": 0.0,
    "cat__State_MN": 0.0,
    "cat__State_MO": 0.0,
    "cat__State_MS": 0.0,
    "cat__State_NC": 0.0,
    "cat__State_NE": 0.0,
    "cat__State_NH": 0.0,
    "cat__State_NJ": 0.0,
    "cat__State_NM": 0.0,
    "cat__State_NV": 0.0,
    "cat__State_NY": 0.0,
    "cat__State_OH": 0.0,
    "cat__State_OK": 0.0,
    "cat__State_OR": 0.0,
    "cat__State_PA": 0.0,
    "cat__State_RI": 0.0,
    "cat__State_SC": 0.0,
    "cat__State_TN": 0.0,
    "cat__State_TX": 0.0,
    "cat__State_UT": 0.0,
    "cat__State_VA": 0.0,
    "cat__State_VT": 0.0,
    "cat__State_WA": 0.0,
    "cat__State_WI": 0.0,
    "cat__State_WV": 0.0,
    "cat__Wind_Direction_Group_CALM": 0.0,
    "cat__Wind_Direction_Group_E": 0.0,
    "cat__Wind_Direction_Group_N": 0.0,
    "cat__Wind_Direction_Group_NE": 0.0,
    "cat__Wind_Direction_Group_NW": 0.0,
    "cat__Wind_Direction_Group_S": 0.0,
    "cat__Wind_Direction_Group_SE": 0.0,
    "cat__Wind_Direction_Group_SW": 1.0,
    "cat__Wind_Direction_Group_UNKNOWN": 0.0,
    "cat__Wind_Direction_Group_VARIABLE": 0.0,
    "cat__Wind_Direction_Group_W": 0.0,
    "cat__Weather_Group_Clear": 1.0,
    "cat__Weather_Group_Cloudy": 0.0,
    "cat__Weather_Group_Fog_Haze": 0.0,
    "cat__Weather_Group_Other": 0.0,
    "cat__Weather_Group_Rain": 0.0,
    "cat__Weather_Group_Snow_Ice": 0.0,
    "cat__Weather_Group_Storm": 0.0,
    "cat__Weather_Group_Unknown": 0.0,
}

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
    features: Dict[str, float] = Field(..., examples=[EXAMPLE_FEATURES])  # pre-fills /docs


@app.get("/")
def root():
    return {
        "message": "Road Accident Severity API is running.",
        "try_example_directly": "/predict/example",
        "interactive_docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "n_features": len(feature_columns), "thresholds": thresholds}


@app.get("/features")
def features():
    """Lists the exact column names the model expects."""
    return {"feature_columns": feature_columns}


def run_prediction(features: Dict[str, float]):
    unknown = [c for c in features if c not in feature_columns]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown feature(s): {unknown}. See GET /features.")

    # Build one row in the same column order as training; columns not sent are filled with 0
    row = pd.DataFrame([features]).reindex(columns=feature_columns, fill_value=0)

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


@app.post("/predict")
def predict(data: AccidentInput):
    """Predict for custom input. The example features are pre-filled - just press Execute."""
    return run_prediction(data.features)


@app.get("/predict/example")
def predict_example():
    """One click: runs the model on the built-in example features (no input needed)."""
    return {"input_features": EXAMPLE_FEATURES, **run_prediction(EXAMPLE_FEATURES)}
