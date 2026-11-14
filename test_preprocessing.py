
import pandas as pd

from utils.preprocessing import build_preprocessor

# Load cleaned dataset
df = pd.read_csv("data/cleaned_salary_data.csv")

# Separate input and target
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Build preprocessing object
preprocessor = build_preprocessor()

# Transform features for this preprocessing test
X_transformed = preprocessor.fit_transform(X)

print("Original feature shape:", X.shape)

print("Transformed feature shape:", X_transformed.shape)

print("\nPreprocessing completed successfully!")

print("\nFirst 5 transformed rows:")
print(X_transformed[:5])
