from fastapi import FastAPI
import pandas as pd
import pickle
import os

app = FastAPI()

# Load trained model
MODEL_PATH = "models/model.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None


@app.get("/")
def home():
    return {"message": "MLOps Pipeline API is running 🚀"}


@app.post("/predict")
def predict(area: float, bedrooms: int, bathrooms: int):
    if model is None:
        return {"error": "Model not found. Train the model first."}

    # Create dataframe for prediction
    input_data = pd.DataFrame(
        [[area, bedrooms, bathrooms]],
        columns=["area", "bedrooms", "bathrooms"]
    )

    prediction = model.predict(input_data)[0]

    return {
        "predicted_price": float(prediction)
    }

@app.get("/")
def read_root():
    return {"message": "MLOps Pipeline Running"}

@app.get("/health")
def health():
    return {"status": "ok"}