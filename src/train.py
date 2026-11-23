"""
Telecom Customer Churn Model Training Script
--------------------------------------------
This script:
1. Loads the telecom customer dataset (`data/churn_data.csv`).
2. Splits the data into stratified train and test sets (80/20).
3. Builds a Scikit-Learn Pipeline with:
   - Numerical preprocessing: SimpleImputer (median) + StandardScaler
   - Categorical preprocessing: SimpleImputer (mode) + OneHotEncoder
   - Classifier: RandomForestClassifier
4. Evaluates the model with Accuracy, Precision, Recall, and F1-Score.
5. Saves the complete trained pipeline to `models/churn_model.pkl`.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Resolve file paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "churn_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_model.pkl"

# Define feature columns clearly for readability and consistency
NUMERIC_FEATURES = ["tenure", "monthly_charges", "support_calls"]
CATEGORICAL_FEATURES = ["contract_type", "internet_service"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "churn"


def build_training_pipeline() -> Pipeline:
    """Create a complete Scikit-Learn preprocessing + Random Forest pipeline."""
    # Numerical columns: impute missing values with median, then standardize
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical columns: impute missing values with mode, then one-hot encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine numeric and categorical transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    # Full pipeline: preprocessing followed by Random Forest Classifier
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model_pipeline


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run `python data/generate_data.py` first."
        )

    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {DATA_PATH} ({df.shape[0]:,} rows, {df.shape[1]} columns)")

    # 2. Separate features (X) and binary target (y: Yes=1, No=0)
    X = df[ALL_FEATURES]
    y = (df[TARGET_COL] == "Yes").astype(int)

    # 3. Stratified train/test split keeps the same churn ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")

    # 4. Build and fit the pipeline
    pipeline = build_training_pipeline()
    pipeline.fit(X_train, y_train)

    # 5. Predict on the hold-out test set and calculate metrics
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--- Random Forest Evaluation Metrics (Test Set) ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # 6. Save the trained pipeline to disk
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved successfully to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
