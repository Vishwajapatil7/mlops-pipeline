import os
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression


def train_model():
    # Create models folder if not exists
    os.makedirs("models", exist_ok=True)

    # Load data (tracked by DVC)
    data_path = "data/data.csv"
    df = pd.read_csv(data_path)

    # Features and target
    X = df[["area", "bedrooms", "bathrooms"]]
    y = df["price"]

    # Initialize model
    model = LinearRegression()

    # Train model
    model.fit(X, y)

    # Save trained model
    model_path = "models/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("✅ Model trained successfully!")
    print(f"Model saved at: {model_path}")


if __name__ == "__main__":
    train_model()