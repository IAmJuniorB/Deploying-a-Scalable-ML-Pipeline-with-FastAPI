import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Set the project path and data path
project_path = "C:/Users/beats/Machine_Learning_Project/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)

# Load the census data
data = pd.read_csv(data_path)
print(data.head())

# Split the data into features and target variable
X = data.drop(columns="salary")
y = data["salary"]

# Split the data into 80% training and 20% test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine the features and target into train and test datasets
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the data using the provided function
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model using the training dataset
model = train_model(X_train, y_train)

# Define paths for saving the model and encoder
model_path = os.path.join(project_path, "model", "model.pkl")
encoder_path = os.path.join(project_path, "model", "encoder.pkl")

# Save the model and encoder
save_model(model, model_path)
save_model(encoder, encoder_path)

# Load the model
model = load_model(model_path)

# Run inference on the test dataset
preds = inference(model, X_test)

# Calculate and print the metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")

# Compute the performance on model slices using the performance_on_categorical_slice function
# Iterate through the categorical features
for col in cat_features:
    # Iterate through the unique values in one categorical feature
    for slice_value in sorted(test[col].unique()):
        count = test[test[col] == slice_value].shape[0]
        
        # Calculate performance metrics for each slice
        precision, recall, fbeta = performance_on_categorical_slice(
            test,
            column_name=col,
            slice_value=slice_value,
            categorical_features=cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            model=model,
        )
        
        # Save the output to slice_output.txt
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slice_value}, Count: {count:,}", file=f)
            print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}", file=f)