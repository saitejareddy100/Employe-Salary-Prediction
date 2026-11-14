
import joblib
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_salary_model.pkl"

# Load trained model
model = joblib.load(MODEL_PATH)

print("Salary prediction model loaded successfully!")

# New employee information
employee = pd.DataFrame([{
    "Years of Experience": 5,
    "Education Level": "Bachelor's",
    "Job Title": "Software Engineer"
}])

# Predict salary
predicted_salary = model.predict(employee)[0]

print("\n========== SALARY PREDICTION ==========")
print("Experience: 5 years")
print("Education: Bachelor's")
print("Job Title: Software Engineer")

print("\nPredicted Salary:", round(predicted_salary, 2))
