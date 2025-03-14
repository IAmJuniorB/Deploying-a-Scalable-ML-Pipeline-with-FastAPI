import pytest
# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    # add description for the first test
    """
    # Your code here
    pass


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # add description for the second test
    """
    # Your code here
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    # Your code here
    pass


import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Test if the model uses the expected algorithm
def test_train_model():
    """Test if the model uses the expected algorithm."""
    model = train_model(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
    assert isinstance(model, RandomForestClassifier)

# Test if the computing metrics functions return the expected value
def test_compute_model_metrics():
    """Test if the computing metrics functions return the expected value."""
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == precision_score(y, preds, zero_division=1)
    assert recall == recall_score(y, preds, zero_division=1)
    assert fbeta == fbeta_score(y, preds, beta=1, zero_division=1)

# Test if the inference function returns the expected type of result
def test_inference():
    """Test if the inference function returns the expected type of result."""
    model = train_model(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
    preds = inference(model, np.array([[5, 6], [7, 8]]))
    assert isinstance(preds, np.ndarray)

# Test if the data processing returns the expected type of result
def test_process_data():
    """Test if the data processing returns the expected type of result."""
    import pandas as pd
    data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': ['a', 'b', 'c'],
        'label': [0, 1, 0]
    })
    cat_features = ['feature2']
    X, y, encoder, lb = process_data(data, cat_features, 'label', training=True)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(encoder, type(None)) or isinstance(encoder, object) # OneHotEncoder
    assert isinstance(lb, type(None)) or isinstance(lb, object) # LabelBinarizer

# Test if the save_model function saves the model correctly
def test_save_model():
    """Test if the save_model function saves the model correctly."""
    model = train_model(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
    model_path = "test_model.pkl"
    save_model(model, model_path)
    loaded_model = load_model(model_path)
    assert isinstance(loaded_model, RandomForestClassifier)
    # Clean up
    import os
    os.remove(model_path)

# Test if the performance_on_categorical_slice function returns the expected type of result
def test_performance_on_categorical_slice():
    """Test if the performance_on_categorical_slice function returns the expected type of result."""
    import pandas as pd
    data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': ['a', 'b', 'c'],
        'label': [0, 1, 0]
    })
    cat_features = ['feature2']
    
    # Train an encoder and binarizer
    X_train, y_train, encoder, lb = process_data(data, cat_features, 'label', training=True)
    
    # Train the model with the same number of features as the processed data
    model = train_model(X_train, y_train)
    
    # Use the trained encoder and binarizer
    p, r, fb = performance_on_categorical_slice(data, 'feature2', 'a', cat_features, 'label', encoder, lb, model)
    assert isinstance(p, float)
    assert isinstance(r, float)
    assert isinstance(fb, float)