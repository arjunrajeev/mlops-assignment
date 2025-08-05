import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Load the model
try:
    model = joblib.load("./models/best_model.pkl")
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model file not found.")

@app.post("/predict")
def predict(features: dict):
    try:
        input_df = pd.DataFrame([features])
        prediction = model.predict(input_df)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")