
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "india_salary_data.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== INDIA SALARY DATA ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head().to_string())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nIndian Cities:")
print(df["city"].value_counts().head(20))

print("\nJob Titles:")
print(df["job_title"].nunique())

print("\nSalary Statistics (INR):")
print(df["salary_mid_inr"].describe())

print("\nExperience Bands:")
print(df["experience_band"].value_counts())

print("\n========== INSPECTION COMPLETED ==========")
