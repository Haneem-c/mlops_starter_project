# train_model.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- 1. Data Generation (Simulating a complex dataset) ---
print("Generating dummy data...")
# Set seed for reproducibility (crucial MLOps habit!)
np.random.seed(42) 
# Create a feature X (100 samples)
X = np.random.rand(100, 1) * 10 
# Create a target y: relationship y = 2x + 1 + noise
y = 2 * X + 1 + np.random.randn(100, 1) 

# Convert to DataFrame (good practice for data handling with Pandas)
df = pd.DataFrame({'feature_X': X.flatten(), 'target_y': y.flatten()})
print(f"Data shape: {df.shape}")


# --- 2. Data Splitting ---
# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    df[['feature_X']], df['target_y'], test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]}")


# --- 3. Model Training ---
print("Training Linear Regression model...")
model = LinearRegression()
# Fit the model to the training data
model.fit(X_train, y_train)


# --- 4. Model Evaluation ---
# Make predictions on the unseen test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)


# --- 5. Output Results ---
print("\n--- Model Results ---")
print(f"Intercept (c): {model.intercept_:.2f}")
print(f"Coefficient (m): {model.coef_[0]:.2f}")
print(f"Mean Squared Error (MSE) on Test Set: {mse:.4f}")
print("---------------------")