
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "large_salary_data.csv"

OUTPUT_PATH = BASE_DIR / "data" / "cleaned_salary_data.csv"


# =====================================
# 2. LOAD DATASET
# =====================================

df = pd.read_csv(DATA_PATH)

print("Original Dataset Shape:", df.shape)


# =====================================
# 3. REMOVE DUPLICATES
# =====================================

print("\nDuplicates:", df.duplicated().sum())

df.drop_duplicates(inplace=True)


# =====================================
# 4. SELECT REQUIRED COLUMNS
# =====================================

selected_columns = [
    "Department",
    "Job_Title",
    "Experience_Years",
    "Education_Level",
    "Location",
    "Salary"
]

df = df[selected_columns].copy()


# =====================================
# 5. CONVERT DATA TYPES
# =====================================

df["Salary"] = pd.to_numeric(
    df["Salary"], errors="coerce"
)

df["Experience_Years"] = pd.to_numeric(
    df["Experience_Years"], errors="coerce"
)


# =====================================
# 6. HANDLE MISSING VALUES
# =====================================

# Remove rows without target salary
df.dropna(subset=["Salary"], inplace=True)

# Remove invalid salary values
df = df[df["Salary"] > 0]

# Fill missing numerical values
df["Experience_Years"] = df[
    "Experience_Years"
].fillna(df["Experience_Years"].median())

# Fill missing categorical values
categorical_columns = [
    "Department",
    "Job_Title",
    "Education_Level",
    "Location"
]

for column in categorical_columns:

    df[column] = df[column].fillna(
        df[column].mode()[0]
    )

    df[column] = df[column].astype(str).str.strip()


# =====================================
# 7. FINAL DATA VALIDATION
# =====================================

print("\n========== CLEANED DATASET ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nSalary Statistics:")
print(df["Salary"].describe())


# =====================================
# 8. SAVE CLEANED DATASET
# =====================================

df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved successfully!")
print("File:", OUTPUT_PATH)


# =====================================
# 9. SALARY DISTRIBUTION
# =====================================

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="Salary",
    bins=30,
    kde=True
)

plt.title("Employee Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.show()


# =====================================
# 10. EXPERIENCE VS SALARY
# =====================================

plt.figure(figsize=(10, 5))

sns.scatterplot(
    data=df,
    x="Experience_Years",
    y="Salary",
    alpha=0.5
)

plt.title("Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

plt.tight_layout()
plt.show()


# =====================================
# 11. SALARY BY DEPARTMENT
# =====================================

department_salary = (
    df.groupby("Department")["Salary"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(11, 5))

sns.barplot(
    x=department_salary.values,
    y=department_salary.index
)

plt.title("Average Salary by Department")
plt.xlabel("Average Salary")
plt.ylabel("Department")

plt.tight_layout()
plt.show()


# =====================================
# 12. SALARY BY LOCATION
# =====================================

location_salary = (
    df.groupby("Location")["Salary"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 5))

sns.barplot(
    x=location_salary.values,
    y=location_salary.index
)

plt.title("Average Salary by Location")
plt.xlabel("Average Salary")
plt.ylabel("Location")

plt.tight_layout()
plt.show()


print("\nDATA CLEANING AND EDA COMPLETED!")
