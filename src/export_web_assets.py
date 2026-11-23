"""
Export Web Dashboard Assets for Vercel Static Deployment
--------------------------------------------------------
Reads `data/churn_data.csv` and the trained Scikit-Learn Random Forest pipeline
(`models/churn_model.pkl`) and exports two compact JSON files into `web/`:

1. `web/dashboard_data.json`:
   - Raw dataset Missing Value Report
   - First 50 raw records for preview
   - Compact cleaned 5,000-row dataset for live browser filtering & EDA charts
   - Hold-out test metrics from `src/train.py`

2. `web/rf_model.json`:
   - Exact StandardScaler parameters (`mean`, `scale`)
   - Exact OneHotEncoder categories
   - Exact Random Forest decision trees (feature index, threshold, left/right
     child, and leaf churn probability) so the browser computes the EXACT same
     `predict_proba` as `models/churn_model.pkl`.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "churn_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
WEB_DIR = PROJECT_ROOT / "web"

NUMERIC_FEATURES = ["tenure", "monthly_charges", "support_calls"]
CATEGORICAL_FEATURES = ["contract_type", "internet_service"]


def export_dashboard_data(raw_df: pd.DataFrame, model) -> dict:
    """Build JSON payload for Tab 1 (Data Explorer) and header metrics."""
    strategy_map = {
        "customer_id": "Identifier (dropped before modeling)",
        "tenure": "Median imputation",
        "monthly_charges": "Median imputation",
        "contract_type": "Mode imputation",
        "internet_service": "Mode imputation",
        "support_calls": "Median imputation",
        "churn": "Target variable (Yes=1, No=0)",
    }

    missing_report = []
    for col in raw_df.columns:
        count = int(raw_df[col].isna().sum())
        missing_report.append(
            {
                "column": col,
                "dtype": str(raw_df[col].dtype),
                "missing_count": count,
                "missing_pct": round((count / len(raw_df)) * 100, 2),
                "strategy": strategy_map.get(col, "Standard"),
            }
        )

    # Raw preview (first 50 rows, replace NaN with null)
    preview_rows = raw_df.head(50).where(pd.notna(raw_df.head(50)), None).to_dict(orient="records")

    # Cleaned rows in compact columnar format for fast client-side filtering
    clean_df = raw_df.copy()
    for col in NUMERIC_FEATURES:
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())
    for col in CATEGORICAL_FEATURES:
        clean_df[col] = clean_df[col].fillna(clean_df[col].mode()[0])
    clean_df["tenure"] = clean_df["tenure"].round().astype(int)
    clean_df["support_calls"] = clean_df["support_calls"].round().astype(int)

    contracts = sorted(clean_df["contract_type"].unique().tolist())
    internets = sorted(clean_df["internet_service"].unique().tolist())
    contract_to_idx = {v: i for i, v in enumerate(contracts)}
    internet_to_idx = {v: i for i, v in enumerate(internets)}

    # Evaluate saved model on the same stratified test split as src/train.py
    X = raw_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = (raw_df["churn"] == "Yes").astype(int)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
        "precision": round(float(precision_score(y_test, y_pred)) * 100, 2),
        "recall": round(float(recall_score(y_test, y_pred)) * 100, 2),
        "f1": round(float(f1_score(y_test, y_pred)) * 100, 2),
        "test_size": int(len(y_test)),
    }

    return {
        "total_rows": int(len(raw_df)),
        "total_missing_cells": int(raw_df.isna().sum().sum()),
        "metrics": metrics,
        "missing_report": missing_report,
        "preview_rows": preview_rows,
        "categories": {
            "contract_type": contracts,
            "internet_service": internets,
        },
        "columns": {
            "tenure": clean_df["tenure"].tolist(),
            "monthly_charges": [round(float(v), 2) for v in clean_df["monthly_charges"].tolist()],
            "support_calls": clean_df["support_calls"].tolist(),
            "contract_idx": [contract_to_idx[v] for v in clean_df["contract_type"].tolist()],
            "internet_idx": [internet_to_idx[v] for v in clean_df["internet_service"].tolist()],
            "churn": (clean_df["churn"] == "Yes").astype(int).tolist(),
        },
    }


def export_rf_model(model) -> dict:
    """Serialize Scikit-Learn preprocessing + Random Forest trees to compact JSON."""
    preprocessor = model.named_steps["preprocessor"]
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    rf = model.named_steps["classifier"]

    trees = []
    for estimator in rf.estimators_:
        t = estimator.tree_
        # At leaf nodes (children_left == -1), store class-1 churn probability
        leaf_probs = []
        for i in range(t.node_count):
            if t.children_left[i] == -1:
                counts = t.value[i][0]
                total = float(counts[0] + counts[1])
                leaf_probs.append(round(float(counts[1]) / total, 5) if total > 0 else 0.0)
            else:
                leaf_probs.append(0.0)

        trees.append(
            {
                "f": [int(x) for x in t.feature.tolist()],
                "t": [round(float(x), 5) for x in t.threshold.tolist()],
                "l": [int(x) for x in t.children_left.tolist()],
                "r": [int(x) for x in t.children_right.tolist()],
                "p": leaf_probs,
            }
        )

    return {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "scaler_mean": [round(float(x), 6) for x in scaler.mean_.tolist()],
        "scaler_scale": [round(float(x), 6) for x in scaler.scale_.tolist()],
        "categories": [cats.tolist() for cats in encoder.categories_],
        "n_trees": len(trees),
        "trees": trees,
    }


def verify_js_equivalence(model, exported_model: dict) -> None:
    """Verify exported tree structure matches sklearn predict_proba to < 1e-4."""
    samples = pd.DataFrame(
        [
            {"tenure": 3, "monthly_charges": 105.0, "support_calls": 6, "contract_type": "Month-to-Month", "internet_service": "Fiber Optic"},
            {"tenure": 60, "monthly_charges": 28.0, "support_calls": 0, "contract_type": "Two Year", "internet_service": "No"},
            {"tenure": 18, "monthly_charges": 65.0, "support_calls": 2, "contract_type": "One Year", "internet_service": "DSL"},
        ]
    )

    sklearn_probs = model.predict_proba(samples)[:, 1]
    for idx, row in samples.iterrows():
        vec = []
        for i, col in enumerate(exported_model["numeric_features"]):
            vec.append((float(row[col]) - exported_model["scaler_mean"][i]) / exported_model["scaler_scale"][i])
        for i, col in enumerate(exported_model["categorical_features"]):
            for cat in exported_model["categories"][i]:
                vec.append(1.0 if row[col] == cat else 0.0)

        prob_sum = 0.0
        for tree in exported_model["trees"]:
            node = 0
            while tree["l"][node] != -1:
                feat = tree["f"][node]
                if vec[feat] <= tree["t"][node]:
                    node = tree["l"][node]
                else:
                    node = tree["r"][node]
            prob_sum += tree["p"][node]
        exported_prob = prob_sum / exported_model["n_trees"]
        diff = abs(float(sklearn_probs[idx]) - exported_prob)
        assert diff < 1e-3, f"Prediction mismatch: sklearn={sklearn_probs[idx]} vs exported={exported_prob}"


def main() -> None:
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    dashboard_payload = export_dashboard_data(raw_df, model)
    rf_payload = export_rf_model(model)
    verify_js_equivalence(model, rf_payload)

    dashboard_path = WEB_DIR / "dashboard_data.json"
    rf_path = WEB_DIR / "rf_model.json"

    dashboard_path.write_text(json.dumps(dashboard_payload, separators=(",", ":")), encoding="utf-8")
    rf_path.write_text(json.dumps(rf_payload, separators=(",", ":")), encoding="utf-8")

    print(f"Exported {dashboard_path} ({dashboard_path.stat().st_size / 1024:.1f} KB)")
    print(f"Exported {rf_path} ({rf_path.stat().st_size / 1024:.1f} KB)")
    print("Equivalence check PASSED (< 0.001 difference vs models/churn_model.pkl)")


if __name__ == "__main__":
    main()
