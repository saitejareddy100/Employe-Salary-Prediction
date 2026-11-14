
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor():

    # Numerical features
    numerical_features = [
        "Experience_Years"
    ]

    # Categorical features
    categorical_features = [
        "Department",
        "Job_Title",
        "Education_Level",
        "Location"
    ]

    # Numerical preprocessing
    numerical_transformer = StandardScaler()

    # Categorical preprocessing
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_transformer,
                numerical_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    return preprocessor
