# src/data_utils.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # <--- This import is key!

def load_and_split_data(n_samples=100, test_size=0.2, random_state=42):
    """Generates dummy data and splits it for training and testing."""
    
    # 1. Data Generation 
    np.random.seed(random_state) 
    X = np.random.rand(n_samples, 1) * 10 
    y = 2 * X + 1 + np.random.randn(n_samples, 1) 
    df = pd.DataFrame({'feature_X': X.flatten(), 'target_y': y.flatten()})

    # 2. Data Splitting <--- THIS SECTION MUST BE PRESENT!
    # This line CREATES the variables X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(
        df[['feature_X']], df['target_y'], test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test # <--- Now these variables exist!

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data()
    print(f"Loaded and split data. Training size: {X_train.shape[0]}")