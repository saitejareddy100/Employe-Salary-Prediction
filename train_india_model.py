
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 1. Load cleaned Indian dataset
# --------------------------------------------------

DATA_FILE = Path("data/india_salary_cleaned.csv")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Cleaned dataset not found. Run clean_india_data.py first."
    )

df = pd.read_csv(DATA_FILE)

print("Indian dataset loaded:", df.shape)

# --------------------------------------------------
# 2. Select input features and target
# --------------------------------------------------

features = [
    "job_title",
    "role_family",
    "seniority",
    "experience_band",
    "city"
]

target = "salary_mid_inr"

df = df.dropna(subset=[target]).copy()

X = df[features].fillna("Unknown").astype(str)
y = pd.to_numeric(df[target], errors="coerce")

valid_rows = y.notna() & (y > 0)
X = X.loc[valid_rows]
y = y.loc[valid_rows]

print("Records available for training:", len(X))

# --------------------------------------------------
# 3. Split training and testing data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            features
        )
    ]
)

# --------------------------------------------------
# 5. Define models
# --------------------------------------------------

models = {
    "Ridge Regression": Ridge(alpha=10.0),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=25,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

results = []
best_model = None
best_model_name = None
best_r2 = float("-inf")

# --------------------------------------------------
# 6. Train and evaluate models
# --------------------------------------------------

for name, estimator in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE_INR": mae,
        "RMSE_INR": rmse,
        "R2_Score": r2
    })

    print(f"MAE:  ₹{mae:,.2f}")
    print(f"RMSE: ₹{rmse:,.2f}")
    print(f"R2:   {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline
        best_model_name = name

# --------------------------------------------------
# 7. Save best model
# --------------------------------------------------

model_path = MODEL_DIR / "india_salary_model.pkl"

joblib.dump(best_model, model_path)

# Save model evaluation results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="R2_Score",
    ascending=False
)

results_df.to_csv(
    MODEL_DIR / "india_model_results.csv",
    index=False
)

# --------------------------------------------------
# 8. Display final results
# --------------------------------------------------

print("\n========== INDIA MODEL RESULTS ==========")
print(results_df.to_string(index=False))

print("\nBest model:", best_model_name)
print("Best R2 score:", round(best_r2, 4))
print("Model saved to:", model_path)
print("Results saved to: models/india_model_results.csv")

print("\nIndian salary model training completed!")
