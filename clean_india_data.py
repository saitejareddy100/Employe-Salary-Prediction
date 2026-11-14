
import pandas as pd
from pathlib import Path

# File paths
INPUT_FILE = Path("data/india_salary_data.csv")
OUTPUT_FILE = Path("data/india_salary_cleaned.csv")

print("Loading Indian salary dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)

# Keep the columns useful for salary prediction
df = df[
    [
        "job_title",
        "role_family",
        "seniority",
        "experience_band",
        "city",
        "salary_mid_inr"
    ]
].copy()

# Remove records without a valid target salary
df = df.dropna(subset=["salary_mid_inr"])

# Remove invalid or non-positive salaries
df = df[df["salary_mid_inr"] > 0]

# Fill missing categorical values
categorical_columns = [
    "job_title",
    "role_family",
    "seniority",
    "experience_band",
    "city"
]

for column in categorical_columns:
    df[column] = df[column].fillna("Unknown").astype(str).str.strip()

# Remove empty job titles
df = df[df["job_title"] != ""]

# Remove duplicate records
df = df.drop_duplicates()

# Save cleaned dataset
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\n========== INDIA DATA CLEANING ==========")
print("Cleaned dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nSalary statistics (INR):")
print(df["salary_mid_inr"].describe())
print("\nSaved to:", OUTPUT_FILE)
print("\nCleaning completed successfully!")
