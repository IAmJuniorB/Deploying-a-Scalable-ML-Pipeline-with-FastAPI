import pandas as pd

df = pd.read_csv("data/census.csv")

print(df.head())

print(df.info())

print(df.isnull().sum())

for col in df.columns:
    print(f"{col}: {df[col].unique()[:5]}")  # Show first 5 unique values
