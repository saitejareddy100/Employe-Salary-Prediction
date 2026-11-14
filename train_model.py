
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from utils.preprocessing import build_preprocessor


# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "cleaned_salary_data.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# =====================================
# 2. LOAD DATASET
# =====================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# =====================================
# 3. FEATURES AND TARGET
# =====================================

features = [
    "Experience_Years",
    "Department",
    "Job_Title",
    "Education_Level",
    "Location"
]

X = df[features]
y = df["Salary"]


# =====================================
# 4. TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# =====================================
# 5. DEFINE MODELS
# =====================================

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=12,
        min_samples_leaf=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
}


# =====================================
# 6. TRAIN AND EVALUATE
# =====================================

results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    r2 = r2_score(y_test, predictions)

    mae = mean_absolute_error(
        y_test, predictions
    )

    rmse = mean_squared_error(
        y_test, predictions
    ) ** 0.5

    results.append({
        "Model": name,
        "R2 Score": r2,
        "MAE": mae,
        "RMSE": rmse
    })

    trained_models[name] = pipeline

    print("R2 Score:", round(r2, 4))
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))


# =====================================
# 7. MODEL COMPARISON
# =====================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========")

print(results_df.to_string(index=False))


# =====================================
# 8. SELECT BEST MODEL
# =====================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)


# =====================================
# 9. SAVE MODEL
# =====================================

model_path = MODEL_DIR / "best_salary_model.pkl"

joblib.dump(best_model, model_path)

results_path = MODEL_DIR / "model_results.csv"

results_df.to_csv(results_path, index=False)

print("\nBest model saved:", model_path)
print("Results saved:", results_path)

print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")
