import pandas as pd
import pickle
import yaml
from sklearn.ensemble import RandomForestRegressor

# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

n_estimators = params["train"]["n_estimators"]
random_state = params["train"]["random_state"]

# Load dataset
data = pd.read_csv("data/data.csv")

# Define features and target
X = data.drop("price", axis=1)
y = data["price"]

# Train regression model
model = RandomForestRegressor(
    n_estimators=n_estimators,
    random_state=random_state
)

model.fit(X, y)

# Save model
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Regression model trained successfully!")