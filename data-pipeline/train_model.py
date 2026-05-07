import json
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "insurance_fraud_dataset.xlsx"
# Trained artifacts go under ml-service-python/artifacts/ (see INSUREGUARD_ARTIFACT_DIR).
ARTIFACT_DIR = ROOT_DIR.parent / "ml-service-python" / "artifacts"

# Honest offline eval: chronological holdout ("time") vs no shared customers ("group")
EVAL_SPLIT: Literal["time", "group"] = "time"
TRAIN_FRAC = 0.8

# Diagnostics: raw columns to drop only for ablation study (printed, not persisted to ML service)
ABLATION_COLUMNS = ["total_claim_amount", "injury_claim"]

id_cols = ["claim_id", "customer_id"]

leakage_cols = [
    "total_fraud_claims",
    "total_claims_by_customer",
    "avg_claim_amount",
    "max_claim_amount",
]

high_cardinality_cols = [
    "policy_number",
    "auto_model",
]

DROP_COLS = list(dict.fromkeys(id_cols + leakage_cols + high_cardinality_cols))

# Used only for ordering the time split — excluded from model inputs (avoid exact-date leakage)
CLAIM_DATE_SPLIT_COL = "claim_date"

HIGH_CARD_UNIQUE_THRESHOLD = 20
TOP_CATEGORY_LEVELS = 15


def print_separator_title(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_label_separability(df_labeled: pd.DataFrame, amount_cols: list[str]) -> None:
    """If fraud/non-fraud differ a lot on $ fields, curves read as trivially separable."""
    present = [c for c in amount_cols if c in df_labeled.columns]
    if not present:
        return
    print_separator_title("Label separability (mean $ features by fraud_reported)")
    tab = df_labeled.groupby("fraud_reported", observed=False)[present].mean(numeric_only=True)
    print(tab)
    print(
        "(Large gaps here often explain near-perfect AUROC/accuracy on synthetic-ish sheets,"
        "\neven after honest time/group splits.)"
    )


def print_feature_importances(model: XGBClassifier, feature_names: list[str], top_k: int = 20) -> None:
    fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print_separator_title(f"Top {top_k} XGBoost feature_importances (impurity-based)")
    print(fi.head(top_k).to_string())


def print_fraud_corr_sanity(df_labeled: pd.DataFrame, fraud_encoded: pd.Series) -> None:
    """Warn on very strong numeric correlation with fraud (possible leakage)."""
    mask = fraud_encoded.notna()
    numeric = df_labeled.loc[mask].select_dtypes(include=["number"]).copy()
    if numeric.shape[1] == 0:
        return

    probe = numeric.assign(_fraud=fraud_encoded.loc[mask].astype(float).values)

    corr = probe.corr(numeric_only=True)["_fraud"].drop(labels=["_fraud"], errors="ignore")
    corr = corr.dropna().sort_values(ascending=False)

    print("\nTop numeric correlations vs fraud_reported (Y=1,N=0) -- pre leakage drop:")
    print(corr.head(10))

    leakage_risk = corr[corr.abs() >= 0.9]
    if len(leakage_risk):
        print(
            "\nWARNING: correlations with |r| >= 0.9 detected -- inspect for leakage:",
            leakage_risk.to_dict(),
            sep="\n",
        )


def _fit_cat_caps(train: pd.DataFrame) -> dict[str, set[Any]]:
    cap_state: dict[str, set[Any]] = {}
    text_cols = train.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in text_cols:
        nunq = train[col].nunique(dropna=True)
        if nunq <= HIGH_CARD_UNIQUE_THRESHOLD:
            continue
        top_values = train[col].dropna().value_counts().nlargest(TOP_CATEGORY_LEVELS).index.tolist()
        cap_state[col] = set(top_values)
    return cap_state


def _apply_cat_caps(frame: pd.DataFrame, cap_state: dict[str, set[Any]]) -> pd.DataFrame:
    out = frame.copy()
    for col, top_set in cap_state.items():
        if col not in out.columns:
            continue

        def _buck(v):  # noqa: ANN001
            if pd.isna(v):
                return v
            return v if v in top_set else "OTHER"

        out[col] = out[col].map(_buck)
    return out


def _fit_imputer(train: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = train.select_dtypes(include=["number"]).columns.tolist()
    meds_series = train[numeric_cols].median(numeric_only=True).fillna(0)

    modes: dict[str, Any] = {}
    for col in train.columns.difference(numeric_cols):
        m = train[col].dropna().mode()
        modes[col] = m.iloc[0] if len(m) else ""

    return {
        "numeric_cols": numeric_cols,
        "medians": meds_series.to_dict(),
        "modes": modes,
    }


def _apply_imputer(frame: pd.DataFrame, state: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    for col in state["numeric_cols"]:
        if col not in out.columns:
            continue
        med = state["medians"].get(col, 0)
        if pd.isna(med):
            med = 0
        out[col] = out[col].fillna(med)

    for col, mode_val in state["modes"].items():
        if col not in out.columns:
            continue
        out[col] = out[col].fillna(mode_val)

    return out


def prepare_train_test_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_drop: list[str],
    extra_drop: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Train-only caps + imputer, shared dummy schema.
    Returns: X_train (encoded), X_test (encoded), X_train_imp (pre-dummies), X_test_imp, dummy_cols
    """
    extra = extra_drop or []

    X_train_raw = train_df.drop(columns=feature_drop, errors="ignore")
    X_train_raw = X_train_raw.drop(columns=DROP_COLS, errors="ignore")
    X_train_raw = X_train_raw.drop(columns=extra, errors="ignore")

    X_test_raw = test_df.drop(columns=feature_drop, errors="ignore")
    X_test_raw = X_test_raw.drop(columns=DROP_COLS, errors="ignore")
    X_test_raw = X_test_raw.drop(columns=extra, errors="ignore")

    cap_state = _fit_cat_caps(X_train_raw)
    X_train_cap = _apply_cat_caps(X_train_raw, cap_state)
    imputer_state = _fit_imputer(X_train_cap)
    X_train_imp = _apply_imputer(X_train_cap, imputer_state)

    X_test_cap = _apply_cat_caps(X_test_raw, cap_state)
    X_test_imp = _apply_imputer(X_test_cap, imputer_state)

    X_train_enc = pd.get_dummies(X_train_imp, drop_first=True)
    dummy_cols = X_train_enc.columns.tolist()

    X_test_enc = pd.get_dummies(X_test_imp, drop_first=True)
    X_test_enc = X_test_enc.reindex(columns=dummy_cols, fill_value=0)

    return X_train_enc, X_test_enc, X_train_imp, X_test_imp, dummy_cols


def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out[CLAIM_DATE_SPLIT_COL] = pd.to_datetime(
        out[CLAIM_DATE_SPLIT_COL],
        errors="coerce",
    )
    out = out.sort_values(CLAIM_DATE_SPLIT_COL, na_position="last")

    n = len(out)
    split_index = max(1, min(int(n * TRAIN_FRAC), n - 1))
    train_df = out.iloc[:split_index].copy()
    test_df = out.iloc[split_index:].copy()

    ct = train_df[CLAIM_DATE_SPLIT_COL].dropna()
    ce = test_df[CLAIM_DATE_SPLIT_COL].dropna()
    if len(ct) and len(ce):
        print(
            f"Time split: train max claim_date ~= {pd.Timestamp(ct.max()).date()} "
            f"({len(train_df)} rows)"
        )
        print(
            f"            test min claim_date ~= {pd.Timestamp(ce.min()).date()} "
            f"({len(test_df)} rows)"
        )
    else:
        print(f"Time split (NaT-heavy dates): train {len(train_df)} | test {len(test_df)} rows")

    return train_df, test_df


def group_split(df: pd.DataFrame, y: pd.Series, groups: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    gss = GroupShuffleSplit(n_splits=1, test_size=1.0 - TRAIN_FRAC, random_state=42)
    train_idx, test_idx = next(gss.split(df, y, groups=groups))

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    print(
        f"Group split (customer_id): train {len(train_df)} rows ({train_df['customer_id'].nunique()} customers)"
    )
    print(
        f"                          test {len(test_df)} rows ({test_df['customer_id'].nunique()} customers)"
    )
    return train_df, test_df


def _xgb_clone() -> XGBClassifier:
    return XGBClassifier(
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=6,
        subsample=0.78,
        colsample_bytree=0.62,
        reg_alpha=1.8,
        reg_lambda=5.0,
        gamma=0.5,
        eval_metric="logloss",
    )


df = pd.read_excel(
    DATA_PATH,
    sheet_name="Claims_Full",
)
excel_shape = df.shape
print("Dataset loaded:", excel_shape)

df.replace("?", pd.NA, inplace=True)
print(f"Before cleaning: {excel_shape}")

df = df.dropna(subset=["fraud_reported"])
y_labels = df["fraud_reported"].map({"Y": 1, "N": 0})
bad_label = y_labels.isna()
if bad_label.any():
    df = df.loc[~bad_label].copy()
    y_labels = y_labels.loc[~bad_label]

print_fraud_corr_sanity(df, y_labels.astype(float))

print_label_separability(
    df,
    amount_cols=[
        "total_claim_amount",
        "injury_claim",
        "vehicle_claim",
        "property_claim",
    ],
)

if EVAL_SPLIT == "time":
    print(f"\nEval split: TIME-BASED ({TRAIN_FRAC:.0%} earliest -> train)")
    train_df, test_df = time_based_split(df)
elif EVAL_SPLIT == "group":
    print(f"\nEval split: GROUP BY customer_id (~{TRAIN_FRAC:.0%} customers -> train)")
    if "customer_id" not in df.columns:
        raise RuntimeError("customer_id missing; cannot run group split")
    train_df, test_df = group_split(df, y_labels.loc[df.index], df["customer_id"])

y_train = train_df["fraud_reported"].map({"Y": 1, "N": 0})
y_test = test_df["fraud_reported"].map({"Y": 1, "N": 0})

feature_drop = ["fraud_reported"]
if CLAIM_DATE_SPLIT_COL in train_df.columns:
    feature_drop = [*feature_drop, CLAIM_DATE_SPLIT_COL]

X_train, X_test, X_train_imp, X_test_imp, dummy_columns = prepare_train_test_matrices(
    train_df, test_df, feature_drop, extra_drop=[]
)

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(X_train_imp.columns.tolist(), ARTIFACT_DIR / "raw_feature_columns.pkl")

defaults: dict[str, object] = {}
for col in X_train_imp.columns:
    dtype = X_train_imp[col].dtype
    if pd.api.types.is_integer_dtype(dtype):
        defaults[col] = 0
    elif pd.api.types.is_float_dtype(dtype):
        defaults[col] = 0.0
    else:
        defaults[col] = ""
joblib.dump(defaults, ARTIFACT_DIR / "raw_feature_defaults.pkl")

training_stats: dict[str, dict[str, float]] = {}
for col in X_train_imp.columns:
    if not pd.api.types.is_numeric_dtype(X_train_imp[col]):
        continue
    s = pd.to_numeric(X_train_imp[col], errors="coerce").dropna()
    if len(s) == 0:
        continue
    m = float(s.mean())
    stdev = float(s.std(ddof=0))
    if pd.isna(stdev) or stdev != stdev:
        stdev = 0.0
    training_stats[str(col)] = {"mean": round(m, 8), "std": round(stdev, 8)}
training_stats_path = ARTIFACT_DIR / "training_stats.json"
training_stats_path.write_text(json.dumps(training_stats, indent=2), encoding="utf-8")
print(f"Training distribution stats saved: {training_stats_path}")

print(f"\nTrain matrix: {X_train.shape} | Test matrix: {X_test.shape}")

# NOTE: sklearn's XGBClassifier.fit() here does not accept callbacks / early stopping on this wrapper.
model = _xgb_clone()
model.fit(X_train, y_train, verbose=False)

# SHAP service: realistic interventional baseline (encoded matrix = same space as XGBoost input)
SHAP_BG_MAX = min(500, len(X_train))
shap_background = (
    X_train.sample(n=SHAP_BG_MAX, random_state=42) if SHAP_BG_MAX < len(X_train) else X_train.copy()
)
shap_bg_path = ARTIFACT_DIR / "shap_background.parquet"
shap_background.to_parquet(shap_bg_path, index=False)
print(f"SHAP background saved: {shap_bg_path} ({len(shap_background)} x {X_train.shape[1]})")

global_importance = {
    str(feat): float(imp) for feat, imp in zip(dummy_columns, model.feature_importances_.tolist())
}
joblib.dump(global_importance, ARTIFACT_DIR / "global_importance.pkl")

y_pred = model.predict(X_test)
print("\nModel Performance (honest split):")
print(classification_report(y_test, y_pred))
print_feature_importances(model, dummy_columns, top_k=20)

# Ablation on the same temporal / group folds (does not overwrite saved artifacts).
present_ablate = [c for c in ABLATION_COLUMNS if c in df.columns]
if present_ablate:
    print_separator_title(
        "Ablation experiment (same split; drop high-signal amount columns)"
    )
    print(f"Dropped raw columns for this diagnostic only: {present_ablate}")
    X_train_a, X_test_a, _, _, _ = prepare_train_test_matrices(
        train_df, test_df, feature_drop, extra_drop=present_ablate
    )
    m_ab = _xgb_clone()
    m_ab.fit(X_train_a, y_train, verbose=False)
    y_pred_a = m_ab.predict(X_test_a)
    print(f"Matrices: train {X_train_a.shape}, test {X_test_a.shape}")
    print(classification_report(y_test, y_pred_a))
    print(f"Held-out accuracy (full vs ablated): {accuracy_score(y_test, y_pred):.4f} vs {accuracy_score(y_test, y_pred_a):.4f}")

print_separator_title("Documentation note")
print(
    "If metrics stay ~1.0 under time/group splits, treat that as DATA separability,"
    "\nnot a broken pipeline -- keep reporting importances + correlations + ablation like above."
)

fraud_model_path = ARTIFACT_DIR / "fraud_model.pkl"
features_path = ARTIFACT_DIR / "model_features.pkl"

joblib.dump(model, fraud_model_path)
joblib.dump(dummy_columns, features_path)

print("\nModel + features saved!")
print(f"Artifacts (single location): {ARTIFACT_DIR}")
