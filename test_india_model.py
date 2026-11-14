
import joblib
import pandas as pd

model = joblib.load("models/india_salary_model.pkl")

sample_employee = pd.DataFrame([{
    "job_title": "Software Engineer",
    "role_family": "software-eng",
    "seniority": "mid",
    "experience_band": "4-6 yrs",
    "city": "Hyderabad"
}])

predicted_salary = model.predict(sample_employee)[0]

print("\n========== INDIA SALARY PREDICTION ==========")
print("Job Title: Software Engineer")
print("Location: Hyderabad")
print("Experience: 4-6 years")
print(f"Predicted Annual Salary: ₹{predicted_salary:,.0f}")
