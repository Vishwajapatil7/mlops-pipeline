from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle
import os

# Initialize FastAPI app
app = FastAPI(title="MLOps House Price Prediction API")

# Model path
MODEL_PATH = "models/model.pkl"

# Load model at startup
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        print("✅ Model loaded successfully!")
else:
    print("⚠️ Model file not found. Please train the model first.")


# Root endpoint
@app.get("/")
def home():
    return {"message": "MLOps Pipeline API is running 🚀"}


# Health check endpoint (used in Docker & CI/CD)
@app.get("/health")
def health():
    return {"status": "ok"}


# Prediction endpoint
@app.post("/predict")
def predict(area: float, bedrooms: int, bathrooms: int):
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not found. Please train the model first."
        )

    try:
        # Create dataframe with correct feature order
        input_data = pd.DataFrame(
            [[area, bedrooms, bathrooms]],
            columns=["area", "bedrooms", "bathrooms"]
        )

        # Make prediction
        prediction = model.predict(input_data)[0]

        return {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "predicted_price": float(prediction)
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )