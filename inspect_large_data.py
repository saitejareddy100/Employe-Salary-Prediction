
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "large_salary_data.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== LARGE DATASET ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nSalary Statistics:")
print(df["Salary"].describe())

print("\n========== INSPECTION COMPLETED ==========")
