import pandas as pd
import pickle
import json
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("data/data.csv")

# Define features and target (your target column is 'price')
X = data.drop("price", axis=1)
y = data["price"]

# Load trained model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Make predictions
preds = model.predict(X)

# Calculate metrics (compatible with all sklearn versions)
mse = mean_squared_error(y, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y, preds)

# Store metrics in dictionary
metrics = {
    "rmse": float(rmse),
    "r2_score": float(r2)
}

# Save metrics for DVC tracking
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Evaluation completed successfully!")
print("Metrics:", metrics)