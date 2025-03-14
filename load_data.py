import pandas as pd
from ml.data import process_data

data = pd.read_csv('C:\\Users\\beats\\Documents\\Machine_Learning_FastAPI\\Take2\\Deploying-a-Scalable-ML-Pipeline-with-FastAPI\\data\\census.csv')

print(data.head())

# Define the categorical features
categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex"
]

# Define the label column
label = "salary"

# Preprocess the data
X, y, encoder, lb = process_data(
    data, categorical_features=categorical_features, label=label, training=True
)

# Print the preprocessed data and labels
print("Preprocessed X:\n", X)
print("Preprocessed y:\n", y)
