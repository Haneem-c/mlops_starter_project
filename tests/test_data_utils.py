# tests/test_data_utils.py

import numpy as np
from src.data_utils import load_and_split_data

# Test 1: Check if the split ratio is correct (80/20)
def test_split_ratio():
    # Arrange: Run the data function
    X_train, X_test, y_train, y_test = load_and_split_data(n_samples=100)
    
    # Assert (The Rule): Check the number of samples
    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20

# Test 2: Check if the feature dimension is correct (one column)
def test_feature_shape():
    X_train, X_test, _, _ = load_and_split_data(n_samples=50)

    # Assert (The Rule): The feature matrix must have 1 column.
    assert X_train.shape[1] == 1
    assert X_test.shape[1] == 1

# Test 3: Check if the random state works (ensuring reproducibility)
def test_reproducibility():
    # Arrange: Run the function twice with the SAME seed
    X1_train, _, _, _ = load_and_split_data(random_state=1)
    X2_train, _, _, _ = load_and_split_data(random_state=1)
    
    # Assert (The Rule): The two datasets MUST be exactly equal.
    assert np.array_equal(X1_train, X2_train)