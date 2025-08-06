import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from app.logger_config import log
from prometheus_fastapi_instrumentator import Instrumentator
app = FastAPI()
Instrumentator().instrument(app).expose(app)
# Load the model
try:
    model = joblib.load("../models/best_model.pkl")
except FileNotFoundError:
    log.error("Could not find a model to load.")
    raise HTTPException(status_code=500, detail="Model file not found.")

@app.post("/predict")
def predict(features: dict):
    log.info("Received request to predict.")
    try:
        input_df = pd.DataFrame([features])
        prediction = model.predict(input_df)
        log.info(f"Prediction: {prediction}")
        return {"prediction": prediction.tolist()}
    except Exception as e:
        log.error(e)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")