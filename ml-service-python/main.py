from collections import defaultdict
from functools import lru_cache
import hashlib
import threading
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, Query
from pydantic import BaseModel, ConfigDict, Field

_LOG = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent


def _artifact_dir() -> Path:
    """Directory containing model pickles, parquet, training_stats, model_registry (default: ./artifacts)."""
    raw = (os.environ.get("INSUREGUARD_ARTIFACT_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (_ROOT / "artifacts").resolve()


_ARTIFACTS = _artifact_dir()

# Model governance registry (committed artifact; aligns with MODEL_VERSION below).
_REGISTRY_PATH = _ARTIFACTS / "model_registry.json"
with open(_REGISTRY_PATH, encoding="utf-8") as f:
    _MODEL_META: dict[str, Any] = json.load(f)

_TRAINING_STATS_PATH = _ARTIFACTS / "training_stats.json"
_TRAINING_STATS: dict[str, Any] = {}
if _TRAINING_STATS_PATH.exists():
    with open(_TRAINING_STATS_PATH, encoding="utf-8") as f:
        loaded_stats = json.load(f)
        if isinstance(loaded_stats, dict):
            _TRAINING_STATS = loaded_stats

_LOGS_DIR = _ROOT / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)
_PREDICTIONS_LOG_PATH = _LOGS_DIR / "predictions.jsonl"

_SERVE_STARTED = time.perf_counter()

_metrics_lock = threading.Lock()
_log_io_lock = threading.Lock()
_metrics: dict[str, int | float] = {
    "success_count": 0,
    "error_count": 0,
    "latency_sum_ms": 0,
    "explain_latency_sum_ms": 0,
}

_model = joblib.load(_ARTIFACTS / "fraud_model.pkl")
_model_features: list[str] = joblib.load(_ARTIFACTS / "model_features.pkl")
_raw_feature_columns: list[str] = joblib.load(_ARTIFACTS / "raw_feature_columns.pkl")

# Schema hash: SCHEMA_CANON_VERSION labels normalization only (bump when rules change).
# v1: pipe-joined str(c).strip(); v2: json.dumps([str(c).strip], separators=(",", ":")) UTF-8 → SHA-1[:10]
SCHEMA_CANON_VERSION = "v2"
_SCHEMA_CANON = json.dumps([str(c).strip() for c in _model_features], separators=(",", ":"))
_SCHEMA_HASH = hashlib.sha1(_SCHEMA_CANON.encode("utf-8")).hexdigest()[:10]

MODEL_VERSION = "2026-05-06-xgb-shap-v3"
EXPLAIN_VERSION = "shap-prob-intv-v2"
CONTRACT_VERSION = "predict-v3"
PAYLOAD_VERSION = "3.1"
MODEL_FAMILY = "xgboost"

_SHAP_DRIFT_LOGGING = os.environ.get("INSUREGUARD_SHAP_DRIFT_CHECK", "").lower().strip() in (
    "1",
    "true",
    "yes",
    "on",
)


def _nonneg_int_env(name: str, default: str = "0") -> int:
    raw = (os.environ.get(name, default) or default).strip() or default
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _nonneg_float_env(name: str, default: str) -> float:
    raw = (os.environ.get(name, default) or default).strip() or default
    try:
        v = float(raw)
        return v if v >= 0.0 and np.isfinite(v) else float(default)
    except ValueError:
        return float(default)


_EXPLAIN_BUDGET_MS = _nonneg_int_env("INSUREGUARD_EXPLAIN_BUDGET_MS")
_EXPLAIN_OFF = _truthy_env("INSUREGUARD_EXPLAIN_OFF")
_WARN_EXPLAIN_LATENCY_MS = _nonneg_int_env("INSUREGUARD_WARN_EXPLAIN_LATENCY_MS")
_CONFIDENCE_INTERVAL_EPS = _nonneg_float_env("INSUREGUARD_CONFIDENCE_INTERVAL_EPS", "0.04")
_DEBUG_PAYLOAD_VERIFY = _truthy_env("INSUREGUARD_DEBUG_PAYLOAD")


def _model_signature_parts() -> dict[str, Any]:
    tail = MODEL_VERSION.rsplit("-", 1)[-1] if "-" in MODEL_VERSION else MODEL_VERSION
    return {
        "family": MODEL_FAMILY,
        "version": str(tail),
        "features": int(len(_model_features)),
        "schemaHash": str(_SCHEMA_HASH),
        "ciEps": float(round(_CONFIDENCE_INTERVAL_EPS, 8)),
    }


def _model_signature_compact(parts: dict[str, Any] | None = None) -> str:
    p = _model_signature_parts() if parts is None else parts
    ci = f"{round(float(p['ciEps']), 6):g}"
    return f"xgb-{p['version']}|features-{p['features']}|schema-{p['schemaHash']}|ci-{ci}"


def _explain_mode() -> str:
    if _EXPLAIN_OFF:
        return "off"
    if _EXPLAIN_BUDGET_MS > 0:
        return "budgeted"
    return "enabled"


def _clamp_probability(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _confidence_interval_band(center: float, eps: float) -> list[float]:
    raw_low = _clamp_probability(center - eps)
    raw_high = _clamp_probability(center + eps)
    low = float(round(raw_low, 4))
    high = float(round(raw_high, 4))
    return [low, high] if low <= high else [high, low]


_LOGGING_LEVEL_HINT = (
    os.environ.get("LOGGING_LEVEL") or os.environ.get("LOGLEVEL") or "INFO"
).upper()
_SHAP_WARMED = False

_defaults_path = _ARTIFACTS / "raw_feature_defaults.pkl"
_raw_feature_defaults: dict[str, Any] = (
    joblib.load(_defaults_path)
    if _defaults_path.exists()
    else {col: 0 for col in _raw_feature_columns}
)

_global_imp_path = _ARTIFACTS / "global_importance.pkl"
_global_importance: dict[str, float] = {}
if _global_imp_path.exists():
    _global_importance = joblib.load(_global_imp_path)

SHAP_IMPACT_CLIP = 2.0
SHAP_MAX_FEATURES_GUARD = 2000
SHAP_ADDITIVITY_TOLERANCE = 1e-3

# Stable explanationError codes for clients. Emitted today: shap_error, format_error.
# Reserved for future use: input_validation_error, background_mismatch.
_latency_breakdown_mismatch_logged = False

_PREDICT_RESPONSE_KEY_ORDER: tuple[str, ...] = (
    "fraudPrediction",
    "fraudProbability",
    "riskScore",
    "riskLevel",
    "modelConfidence",
    "modelMeta",
    "driftDetected",
    "driftFeatures",
    "confidenceBand",
    "confidenceInterval",
    "confidenceIntervalType",
    "confidenceEpsUsed",
    "reason",
    "reasonSummary",
    "baseValue",
    "approxProbability",
    "approxMatches",
    "approxConsistent",
    "approxDelta",
    "directionScore",
    "explanationStatus",
    "explanationReady",
    "explanationReadyReason",
    "explainAttempted",
    "explainMode",
    "explainBudgetMs",
    "explanationError",
    "explanationSkippedReason",
    "budgetMs",
    "elapsedPreExplainMs",
    "explainSkippedAtMs",
    "budgetUtilization",
    "topFactors",
    "sumTopKPercent",
    "residualPercent",
    "sumTopKImpact",
    "groupTotals",
    "groupCounts",
    "nFeatures",
    "featuresUsed",
    "schemaSize",
    "backgroundSource",
    "nTopK",
    "latency",
    "latencyShare",
    "latencyMs",
    "explainCostMs",
    "explainPolicy",
    "explainTopKRequested",
    "explainTopKUsed",
    "warnings",
    "warningsCount",
    "modelFamily",
    "modelVersion",
    "explainVersion",
    "contractVersion",
    "payloadVersion",
    "schemaHash",
    "schemaCanonVersion",
    "modelSignature",
    "modelSignatureParts",
    "requestTimestamp",
    "timestampSource",
    "requestId",
    "responseSizeBytes",
)


def _ordered_predict_response(blob: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    seen: set[str] = set()
    for k in _PREDICT_RESPONSE_KEY_ORDER:
        if k in blob:
            out[k] = blob[k]
            seen.add(k)
    for ek in sorted(k for k in blob if k not in seen):
        out[ek] = blob[ek]
    return out


def _stable_dedupe_warnings(tags: list[str]) -> list[str]:
    return list(dict.fromkeys(tags))


def _shap_background_source() -> str:
    return "synthetic_zero" if _SHAP_BACKGROUND_FALLBACK_USED else "parquet"


def _safe_out_float(x: Any) -> float:
    xf = float(x)
    return 0.0 if not np.isfinite(xf) else xf


def _sanitize_json_numbers(obj: Any) -> Any:
    if isinstance(obj, float):
        return _safe_out_float(obj)
    if isinstance(obj, np.floating):
        return _safe_out_float(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_json_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json_numbers(v) for v in obj]
    return obj


def _finalize_predict_blob(blob: dict[str, Any]) -> dict[str, Any]:
    def _utf8_len(payload: dict[str, Any]) -> int:
        return len(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
        )

    ordered = _ordered_predict_response(blob)
    work = dict(_sanitize_json_numbers(dict(ordered)))
    work.pop("responseSizeBytes", None)
    work["responseSizeBytes"] = _utf8_len(work)
    for _ in range(24):
        nlen = _utf8_len(work)
        if nlen == work["responseSizeBytes"]:
            break
        work["responseSizeBytes"] = nlen
    out = _ordered_predict_response(work)
    if _DEBUG_PAYLOAD_VERIFY:
        assert int(out["responseSizeBytes"]) == _utf8_len(dict(out))
    return out


def _explanation_ready_reason(
    status: str,
    *,
    explanation_error: str | None,
    skipped_reason: str | None,
) -> str:
    if status == "ok":
        return "ok"
    if status == "fallback":
        return str(explanation_error or "shap_error")
    if status == "disabled":
        if skipped_reason == "budget_exceeded":
            return "budget_exceeded"
        return "explain_disabled"
    return "shap_error"


def _maybe_log_latency_breakdown_mismatch(
    raw_total_ms: int, model_ms: int, explain_ms: int, request_id: str
) -> None:
    global _latency_breakdown_mismatch_logged
    if _latency_breakdown_mismatch_logged:
        return
    summed = int(model_ms) + int(explain_ms)
    if abs(int(raw_total_ms) - summed) <= 1:
        return
    _latency_breakdown_mismatch_logged = True
    _LOG.debug(
        "latency_breakdown_mismatch",
        extra={
            "request_id": request_id,
            "raw_total_ms": raw_total_ms,
            "model_ms": model_ms,
            "explain_ms": explain_ms,
            "summed_ms": summed,
        },
    )


def _shap_fallback_background() -> pd.DataFrame:
    return pd.DataFrame(
        np.zeros((32, len(_model_features)), dtype=np.float64),
        columns=_model_features,
    )


def _resolve_shap_background() -> tuple[pd.DataFrame, bool]:
    """Return (background matrix, True if synthetic fallback substituted for real parquet)."""
    pq = _ARTIFACTS / "shap_background.parquet"
    nfeat = len(_model_features)
    if pq.exists():
        bg = pd.read_parquet(pq).reindex(columns=_model_features, fill_value=0).astype(np.float64)
        if bg.shape[1] == nfeat and len(bg) >= 8:
            return bg, False
    return _shap_fallback_background(), True


_SHAP_BACKGROUND, _SHAP_BACKGROUND_FALLBACK_USED = _resolve_shap_background()
if _SHAP_BACKGROUND.shape[1] != len(_model_features):
    _SHAP_BACKGROUND = _shap_fallback_background()
    _SHAP_BACKGROUND_FALLBACK_USED = True


def _predict_warnings(explain_ms_final: int, explain_attempted_flag: bool) -> list[str]:
    w: list[str] = []
    if _SHAP_BACKGROUND_FALLBACK_USED:
        w.append("background_fallback_used")
    if (
        _WARN_EXPLAIN_LATENCY_MS > 0
        and explain_attempted_flag
        and explain_ms_final >= _WARN_EXPLAIN_LATENCY_MS
    ):
        w.append("high_latency_explain")
    return w


def _baseline_fraud_from_explainer(explainer: Any) -> float:
    """Expected fraud probability implied by SHAP baseline (probability output)."""
    try:
        ev = explainer.expected_value
        flat = np.asarray(ev, dtype=float).ravel()
        if flat.size == 1:
            return float(flat[0])
        cls_list = np.asarray(_model.classes_).tolist()
        if 1 in cls_list:
            return float(flat[cls_list.index(1)])
        return float(flat[-1])
    except Exception:
        return 0.0


_explainer = shap.TreeExplainer(
    _model,
    data=_SHAP_BACKGROUND,
    model_output="probability",
    feature_perturbation="interventional",
)

_SHAP_BASE_VALUE: float = _baseline_fraud_from_explainer(_explainer)


def _warmup_shap() -> None:
    global _SHAP_WARMED
    try:
        dummy = pd.DataFrame(
            np.zeros((1, len(_model_features)), dtype=np.float64),
            columns=_model_features,
        )
        _explainer.shap_values(dummy.astype(np.float64))
        _SHAP_WARMED = True
    except Exception:
        _SHAP_WARMED = False


_warmup_shap()


def _maybe_log_shap_additivity(contrib: np.ndarray, fraud_prob: float, request_id: str) -> None:
    """Cheap check: baseValue + sum(SHAP). Uses the same contrib vector passed in (no recompute)."""
    if not _SHAP_DRIFT_LOGGING:
        return
    approx = _clamp_probability(float(_SHAP_BASE_VALUE + float(contrib.sum())))
    if abs(approx - fraud_prob) > SHAP_ADDITIVITY_TOLERANCE:
        _LOG.debug(
            "shap_drift",
            extra={
                "approx": round(approx, 6),
                "proba": round(float(fraud_prob), 6),
                "request_id": request_id,
            },
        )


def _explain_top_k_requested_and_used(explain: str | None) -> tuple[int, int]:
    """Return (requested semantic k, k after clamping to [1, 10])."""
    if explain is None:
        requested = 5
    else:
        key = str(explain).strip().lower()
        if key == "compact":
            requested = 3
        elif key == "full":
            requested = 10
        elif key.isdigit():
            requested = int(key)
        else:
            requested = 5
    clamped = min(max(int(requested), 1), 10)
    return int(requested), int(clamped)


def _explain_top_k_param(explain: str | None) -> int:
    """Resolve requested explanation depth (clamped [1, 10])."""
    return _explain_top_k_requested_and_used(explain)[1]


def _strict_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None


def _title_segs(segs: list[str]) -> str:
    tiny = frozenset({"of", "and", "or", "by", "for", "to", "in", "per", "as", "vs"})
    tokens: list[str] = []
    for p in segs:
        tokens.extend(str(p).replace("-", " ").split())
    parts: list[str] = []
    for p in tokens:
        low = p.lower()
        if low in tiny:
            parts.append(low)
        elif len(p) <= 2:
            parts.append(p.upper())
        elif low == "zip":
            parts.append(p.upper())
        else:
            parts.append(p.capitalize())
    return " ".join(parts)


def format_feature_name(name: str) -> str:
    """Turn encoded column names into short human-facing labels."""
    segs = name.split("_")
    if not segs:
        return name
    tail = segs[-1].upper()
    if tail in ("YES", "NO"):
        base = "_".join(segs[:-1])
        if not base:
            return tail
        return f"{format_feature_name(base)} ({tail})"
    return _title_segs(segs)


def group_feature(name: str) -> str:
    n = name.lower()
    if "claim" in n:
        return "Claim amounts"
    if n.startswith("incident") or "incident_" in n:
        return "Incident context"
    if any(k in n for k in ("policy", "premium", "umbrella")):
        return "Policy"
    if any(k in n for k in ("customer", "age", "education", "occupation")):
        return "Customer"
    return "Other"


def confidence_band(p: float) -> str:
    if p >= 0.8:
        return "very_high"
    if p >= 0.65:
        return "high"
    if p >= 0.5:
        return "medium"
    if p >= 0.35:
        return "low"
    return "very_low"


def _clip_impact(v: float, lo: float = -SHAP_IMPACT_CLIP, hi: float = SHAP_IMPACT_CLIP) -> float:
    return float(max(min(v, hi), lo))


def _risk_direction(clipped_impact: float, eps: float = 1e-12) -> str:
    if abs(clipped_impact) <= eps:
        return "neutral"
    return "increase_risk" if clipped_impact > 0 else "decrease_risk"


def build_reason_summary(factors: list[dict[str, Any]], fraud_probability: float) -> str:
    if not factors:
        return "Prediction made; explanation unavailable."
    risk = "High" if fraud_probability >= 0.5 else "Low"
    top = factors[:2]
    if len(top) >= 2:
        return (
            f"{risk} fraud risk ({fraud_probability:.2f}) driven mainly by "
            f"{top[0]['feature']} and {top[1]['feature']}."
        )
    return f"{risk} fraud risk ({fraud_probability:.2f}) driven mainly by {top[0]['feature']}."


def _sanitize_encoded_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float64)


def fill_missing_fields(
    data_dict: dict[str, Any],
    required_columns: list[str],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Fill missing keys and None values; ignore keys not in required_columns."""
    out: dict[str, Any] = {}
    for col in required_columns:
        if col in data_dict and data_dict[col] is not None:
            out[col] = data_dict[col]
        else:
            out[col] = defaults.get(col, 0)
    return out


def _row_tuple_for_cache(X: pd.DataFrame) -> tuple[float, ...]:
    vals = np.round(
        X.astype(np.float64).to_numpy(dtype=float, copy=False).ravel(order="C"),
        decimals=6,
    )
    return tuple(float(x) for x in vals)


@lru_cache(maxsize=512)
def _shap_probability_contribs_cached(row_tuple: tuple[float, ...]) -> tuple[float, ...]:
    Xr = pd.DataFrame([list(row_tuple)], columns=_model_features).astype(np.float64)
    raw = _explainer.shap_values(Xr)
    if isinstance(raw, list):
        pos_idx = list(_model.classes_).index(1) if 1 in _model.classes_ else -1
        arr = np.asarray(raw[pos_idx])[0]
    else:
        arr = np.asarray(raw)[0]
    return tuple(float(x) for x in arr)


def _shap_values_single_row(X: pd.DataFrame) -> np.ndarray:
    key = _row_tuple_for_cache(X)
    return np.asarray(_shap_probability_contribs_cached(key), dtype=np.float64)


def explain_top_factors_from_contribs(
    contrib: np.ndarray, top_k: int
) -> tuple[list[dict[str, Any]], float, float, dict[str, float], dict[str, int]]:
    """Returns topFactors, residualPercent, sumTopKPercent, groupTotals, groupCounts."""
    empty_residual = float(round(1.0, 4))
    if len(_model_features) > SHAP_MAX_FEATURES_GUARD:
        return [], empty_residual, 0.0, {}, {}

    c = contrib.astype(float)
    total_all = float(np.abs(c).sum() + 1e-12)

    pairs = sorted(
        zip(_model_features, c.tolist()),
        key=lambda t: abs(float(t[1])),
        reverse=True,
    )
    rows: list[dict[str, Any]] = []
    for f_raw, raw_v in pairs[:top_k]:
        raw_f = float(raw_v)
        clipped = _clip_impact(raw_f)
        pct = abs(raw_f) / total_all
        grp = group_feature(f_raw)
        rows.append(
            {
                "feature": format_feature_name(f_raw),
                "impact": float(clipped),
                "direction": _risk_direction(clipped),
                "globalImportance": float(_global_importance.get(f_raw, 0.0)),
                "percent": float(round(pct, 4)),
                "group": grp,
            }
        )

    rows.sort(key=lambda r: (r["group"], -abs(r["impact"]), r["feature"]))
    for i, item in enumerate(rows, start=1):
        item["rank"] = i

    sum_top_k = float(round(sum(float(it["percent"]) for it in rows), 4))
    residual = float(max(0.0, round(1.0 - sum_top_k, 4)))

    group_totals_collect: defaultdict[str, float] = defaultdict(float)
    group_counts_collect: defaultdict[str, int] = defaultdict(int)
    for item in rows:
        grp = str(item["group"])
        group_totals_collect[grp] += float(item["percent"])
        group_counts_collect[grp] += 1
    group_totals = {k: round(v, 4) for k, v in sorted(group_totals_collect.items())}
    group_counts = {k: int(group_counts_collect[k]) for k in sorted(group_counts_collect)}

    return rows, residual, sum_top_k, group_totals, group_counts


def detect_drift(value: Any, mean: Any, std: Any) -> bool:
    """3-sigma deviation from training marginal (per-feature)."""
    try:
        v = float(value)
        m = float(mean)
        s = float(std)
    except (TypeError, ValueError):
        return False
    if not (np.isfinite(v) and np.isfinite(m) and np.isfinite(s)):
        return False
    if s <= 1e-12:
        return False
    return abs(v - m) > 3 * s


def _evaluate_input_drift(raw_filled: dict[str, Any]) -> tuple[bool, list[str]]:
    if not _TRAINING_STATS:
        return False, []
    drift_feats: list[str] = []
    for feat, stats in _TRAINING_STATS.items():
        if feat not in raw_filled:
            continue
        if not isinstance(stats, dict):
            continue
        if detect_drift(raw_filled[feat], stats.get("mean"), stats.get("std")):
            drift_feats.append(str(feat))
    return (len(drift_feats) > 0, sorted(drift_feats))


def _append_prediction_jsonl(entry: dict[str, Any]) -> None:
    line = json.dumps(entry, separators=(",", ":"), ensure_ascii=False)
    with _log_io_lock:
        with open(_PREDICTIONS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _record_metrics_success(latency_ms: int, explain_ms: int) -> None:
    with _metrics_lock:
        _metrics["success_count"] = int(_metrics["success_count"]) + 1
        _metrics["latency_sum_ms"] = float(_metrics["latency_sum_ms"]) + float(latency_ms)
        _metrics["explain_latency_sum_ms"] = float(_metrics["explain_latency_sum_ms"]) + float(explain_ms)


def _record_metrics_error() -> None:
    with _metrics_lock:
        _metrics["error_count"] = int(_metrics["error_count"]) + 1


def _format_uptime_human(elapsed_s: float) -> str:
    s = max(0, int(elapsed_s))
    h, rem = divmod(s, 3600)
    m, _sec = divmod(rem, 60)
    parts: list[str] = []
    if h:
        parts.append(f"{h}h")
    parts.append(f"{m}m")
    return " ".join(parts) if parts else "0m"


def _predict_impl(raw_claim: dict[str, Any], *, explain: str | None) -> dict[str, Any]:
    t_start = time.perf_counter()
    request_id = str(uuid.uuid4())
    request_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    filled = fill_missing_fields(raw_claim, _raw_feature_columns, _raw_feature_defaults)
    drift_detected, drift_features = _evaluate_input_drift(filled)

    X = _sanitize_encoded_matrix(_encode_row(raw_claim))
    explain_top_k_requested, top_k = _explain_top_k_requested_and_used(explain)

    pred = int(_model.predict(X)[0])
    proba_row = _model.predict_proba(X)[0]
    classes = list(_model.classes_)
    fraud_idx = classes.index(1) if 1 in classes else -1
    fraud_prob = float(proba_row[fraud_idx]) if fraud_idx >= 0 else float(max(proba_row))

    t_after_model = time.perf_counter()
    risk_level = "High" if pred == 1 else "Low"

    top_factors: list[dict[str, Any]] = []
    residual_percent = round(1.0, 4)
    sum_top_k_percent = 0.0
    group_totals: dict[str, float] = {}
    group_counts: dict[str, int] = {}
    reason_summary = "Explanation temporarily unavailable."
    approx_probability: float | None = None
    direction_score: float | None = None
    explanation_status = "fallback"
    explanation_error: str | None = None
    explanation_skipped_reason: str | None = None

    elapsed_pre_explain_ms = int((t_after_model - t_start) * 1000)

    if _EXPLAIN_OFF:
        explanation_status = "disabled"
        explanation_skipped_reason = "explain_disabled"
    elif _EXPLAIN_BUDGET_MS > 0 and elapsed_pre_explain_ms >= _EXPLAIN_BUDGET_MS:
        explanation_status = "disabled"
        explanation_skipped_reason = "budget_exceeded"
    else:
        try:
            contrib = _shap_values_single_row(X)
            direction_score = float(contrib.sum())
            approx_probability = _clamp_probability(float(_SHAP_BASE_VALUE + direction_score))
            _maybe_log_shap_additivity(contrib, fraud_prob, request_id)
            try:
                (
                    top_factors,
                    residual_percent,
                    sum_top_k_percent,
                    group_totals,
                    group_counts,
                ) = explain_top_factors_from_contribs(contrib, top_k=top_k)
                reason_summary = build_reason_summary(top_factors, fraud_prob)
                explanation_status = "ok"
            except Exception:
                explanation_status = "fallback"
                explanation_error = "format_error"
        except Exception:
            explanation_status = "fallback"
            explanation_error = "shap_error"
            direction_score = None
            approx_probability = None

    t_after_explain = time.perf_counter()

    band = confidence_band(fraud_prob)
    sum_top_k_impact = float(sum(float(it["impact"]) for it in top_factors))

    for item in top_factors:
        item["impact"] = float(item["impact"])
        item["globalImportance"] = float(item["globalImportance"])
        item["percent"] = float(item["percent"])
        item["rank"] = int(item["rank"])

    ap = _strict_float(approx_probability)

    t_final = time.perf_counter()
    raw_total_ms = int((t_final - t_start) * 1000)

    model_ms_seg = int((t_after_model - t_start) * 1000)
    explain_ms_seg = int((t_after_explain - t_after_model) * 1000)
    post_ms = int((t_final - t_after_explain) * 1000)

    explain_attempted = explanation_status != "disabled"
    if explain_attempted:
        model_ms_final = int(model_ms_seg)
        explain_ms_final = int(explain_ms_seg + post_ms)
    else:
        model_ms_final = int(model_ms_seg + post_ms)
        explain_ms_final = int(explain_ms_seg)

    latency_ms = int(model_ms_final + explain_ms_final)
    _maybe_log_latency_breakdown_mismatch(raw_total_ms, model_ms_final, explain_ms_final, request_id)

    latency_block = {
        "totalMs": latency_ms,
        "modelMs": int(model_ms_final),
        "explainMs": int(explain_ms_final),
    }
    if latency_ms > 0:
        model_share = round(model_ms_final / float(latency_ms), 4)
        explain_share = round(1.0 - float(model_share), 4)
        latency_share = {"model": float(model_share), "explain": float(explain_share)}
    else:
        latency_share = {"model": 0.0, "explain": 0.0}

    explain_active = explanation_status != "disabled"
    explain_mode = _explain_mode()
    explanation_ready = explanation_status == "ok"
    explanation_ready_reason = _explanation_ready_reason(
        explanation_status,
        explanation_error=explanation_error,
        skipped_reason=explanation_skipped_reason,
    )
    n_features_used = int(len(_model_features))

    explain_policy: dict[str, Any] = {
        "mode": explain_mode,
        "budgetMs": int(max(0, _EXPLAIN_BUDGET_MS)),
    }

    budget_ms_eff = max(0, int(_EXPLAIN_BUDGET_MS))

    warns = _stable_dedupe_warnings(_predict_warnings(explain_ms_final, explain_attempted))
    confidence_eps_used = float(round(_CONFIDENCE_INTERVAL_EPS, 8))
    confidence_interval = _confidence_interval_band(float(fraud_prob), _CONFIDENCE_INTERVAL_EPS)

    sig_parts = _model_signature_parts()

    model_confidence = abs(float(fraud_prob) - 0.5) * 2.0

    blob: dict[str, Any] = {
        "fraudPrediction": int(pred),
        "fraudProbability": float(fraud_prob),
        "riskScore": float(fraud_prob),
        "riskLevel": risk_level,
        "modelConfidence": float(round(model_confidence, 8)),
        "modelMeta": dict(_MODEL_META),
        "driftDetected": bool(drift_detected),
        "driftFeatures": drift_features,
        "confidenceBand": band,
        "confidenceInterval": confidence_interval,
        "confidenceIntervalType": "fixed_epsilon",
        "confidenceEpsUsed": confidence_eps_used,
        "reason": "XGBoost + SHAP (probability scale)",
        "reasonSummary": reason_summary,
        "topFactors": top_factors,
        "sumTopKPercent": float(sum_top_k_percent),
        "residualPercent": float(residual_percent),
        "sumTopKImpact": float(sum_top_k_impact),
        "groupTotals": {k: float(v) for k, v in group_totals.items()},
        "groupCounts": {k: int(v) for k, v in group_counts.items()},
        "nFeatures": n_features_used,
        "featuresUsed": n_features_used,
        "schemaSize": int(n_features_used),
        "backgroundSource": _shap_background_source(),
        "nTopK": int(top_k),
        "latency": latency_block,
        "latencyShare": latency_share,
        "latencyMs": int(latency_ms),
        "explainCostMs": int(latency_block["explainMs"]),  # alias of latency["explainMs"]
        "explainPolicy": explain_policy,
        "explainTopKRequested": int(explain_top_k_requested),
        "explainTopKUsed": int(top_k),
        "warnings": warns,
        "warningsCount": int(len(warns)),
        "modelFamily": MODEL_FAMILY,
        "modelVersion": MODEL_VERSION,
        "explainVersion": EXPLAIN_VERSION,
        "contractVersion": CONTRACT_VERSION,
        "payloadVersion": PAYLOAD_VERSION,
        "schemaHash": _SCHEMA_HASH,
        "schemaCanonVersion": SCHEMA_CANON_VERSION,
        "modelSignature": _model_signature_compact(sig_parts),
        "modelSignatureParts": dict(sig_parts),
        "explanationStatus": explanation_status,
        "explanationReady": bool(explanation_ready),
        "explanationReadyReason": explanation_ready_reason,
        "explainAttempted": bool(explain_attempted),
        "explainMode": explain_mode,
        "explainBudgetMs": int(max(0, _EXPLAIN_BUDGET_MS)),
    }

    if explanation_status == "disabled":
        blob["explainSkippedAtMs"] = int(elapsed_pre_explain_ms)

    if explain_mode == "budgeted" and budget_ms_eff > 0:
        blob["budgetUtilization"] = float(
            round(min(1.0, elapsed_pre_explain_ms / float(budget_ms_eff)), 4)
        )

    if explain_active:
        blob["baseValue"] = float(_SHAP_BASE_VALUE)

    if explanation_status == "fallback" and explanation_error:
        blob["explanationError"] = explanation_error

    if explanation_status == "disabled" and explanation_skipped_reason:
        blob["explanationSkippedReason"] = explanation_skipped_reason
        if explanation_skipped_reason == "budget_exceeded":
            blob["budgetMs"] = int(_EXPLAIN_BUDGET_MS)
            blob["elapsedPreExplainMs"] = int(elapsed_pre_explain_ms)

    # approxMatches / approxConsistent only when approxProbability is present (atomic).
    if explain_active and ap is not None:
        apf = float(ap)
        fp = float(fraud_prob)
        consistent = bool(abs(apf - fp) <= SHAP_ADDITIVITY_TOLERANCE)
        blob["approxProbability"] = apf
        blob["approxMatches"] = consistent
        blob["approxConsistent"] = consistent
        blob["approxDelta"] = float(round(abs(apf - fp), 8))
    if explain_active and direction_score is not None:
        blob["directionScore"] = float(direction_score)

    blob["requestTimestamp"] = request_timestamp
    blob["timestampSource"] = "server_utc"
    blob["requestId"] = request_id

    log_extra: dict[str, Any] = {
        "request_id": request_id,
        "request_timestamp": request_timestamp,
        "fraud_probability": float(fraud_prob),
        "prediction": int(pred),
        "explanation_status": explanation_status,
        "latency_ms": int(latency_ms),
        "features_used": int(n_features_used),
        "schema_hash": str(_SCHEMA_HASH),
        "schema_canon_version": SCHEMA_CANON_VERSION,
        "model_signature": _model_signature_compact(sig_parts),
        "payload_version": PAYLOAD_VERSION,
    }
    if explanation_error:
        log_extra["explanation_error"] = explanation_error
    if explanation_skipped_reason:
        log_extra["explanation_skipped_reason"] = explanation_skipped_reason
    _LOG.info("predict_complete", extra=log_extra)

    return blob


def _finalize_and_record_observability(blob: dict[str, Any]) -> dict[str, Any]:
    finalized = _finalize_predict_blob(blob)
    lat = finalized.get("latency") or {}
    log_entry = {
        "requestId": finalized.get("requestId"),
        "timestamp": finalized.get("requestTimestamp"),
        "riskScore": float(finalized.get("riskScore", 0) or 0),
        "fraudProbability": float(finalized.get("fraudProbability", 0) or 0),
        "latencyMs": int(lat.get("totalMs", finalized.get("latencyMs", 0)) or 0),
    }
    _append_prediction_jsonl(log_entry)
    _record_metrics_success(
        int(lat.get("totalMs", finalized.get("latencyMs", 0)) or 0),
        int(lat.get("explainMs", finalized.get("explainCostMs", 0)) or 0),
    )
    return finalized


app = FastAPI(title="InsureGuard fraud ML")


class ClaimFeatures(BaseModel):
    """Validated claim payload; extras map to categorical / extended raw fields."""

    model_config = ConfigDict(extra="allow")

    total_claim_amount: float | None = Field(default=None, ge=0, le=500_000_000)
    vehicle_claim: float | None = Field(default=None, ge=0, le=50_000_000)
    property_claim: float | None = Field(default=None, ge=0, le=50_000_000)
    injury_claim: float | None = Field(default=None, ge=0, le=50_000_000)
    incident_hour_of_day: int | None = Field(default=None, ge=0, le=23)


class BatchPredictBody(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1, max_length=256)


def _encode_row(raw: dict[str, Any]) -> pd.DataFrame:
    data = fill_missing_fields(raw, _raw_feature_columns, _raw_feature_defaults)

    row = pd.DataFrame([{k: data[k] for k in _raw_feature_columns}])
    row = row.replace("?", pd.NA)

    encoded = pd.get_dummies(row, drop_first=True)
    aligned = encoded.reindex(columns=_model_features, fill_value=0)
    return _sanitize_encoded_matrix(aligned)


@app.post("/predict")
def predict(
    row: ClaimFeatures,
    explain: Annotated[
        str | None,
        Query(
            description="compact|full|numeric k (e.g. 10); requested vs used (clamped [1,10])",
        ),
    ] = None,
) -> dict[str, Any]:
    try:
        raw = row.model_dump(mode="python", exclude_none=True)
        blob = _predict_impl(raw, explain=explain)
        return _finalize_and_record_observability(blob)
    except Exception:
        _record_metrics_error()
        raise


@app.post("/predict/batch")
def predict_batch(
    body: BatchPredictBody,
    explain: Annotated[
        str | None,
        Query(
            description="Same semantics as POST /predict?explain",
        ),
    ] = None,
) -> dict[str, Any]:
    finalized: list[dict[str, Any]] = []
    try:
        for r in body.rows:
            vf = ClaimFeatures.model_validate(r)
            raw = vf.model_dump(mode="python", exclude_none=True)
            blob = _predict_impl(raw, explain=explain)
            finalized.append(_finalize_and_record_observability(blob))
    except Exception:
        _record_metrics_error()
        raise
    return {"results": finalized, "count": len(finalized), "modelMeta": dict(_MODEL_META)}


@app.get("/feature-importance")
def feature_importance(
    top_k: Annotated[int, Query(ge=1, le=500, description="Impurity importance, descending")] = 25,
) -> dict[str, Any]:
    if not _global_importance:
        return {
            "top": [],
            "count": 0,
            "schemaHash": _SCHEMA_HASH,
            "modelMeta": dict(_MODEL_META),
        }
    ranked = sorted(_global_importance.items(), key=lambda kv: kv[1], reverse=True)
    sliced = ranked[:top_k]
    return {
        "top": [{"feature": k, "importance": float(v)} for k, v in sliced],
        "count": len(sliced),
        "schemaHash": _SCHEMA_HASH,
        "modelMeta": dict(_MODEL_META),
    }


@app.get("/metrics")
def service_metrics() -> dict[str, Any]:
    with _metrics_lock:
        succ = int(_metrics["success_count"])
        errs = int(_metrics["error_count"])
        lat_sum = float(_metrics["latency_sum_ms"])
        ex_sum = float(_metrics["explain_latency_sum_ms"])
    avg_lat = round(lat_sum / succ, 4) if succ else 0.0
    avg_expl = round(ex_sum / succ, 4) if succ else 0.0
    return {
        "requestCount": succ,
        "errorCount": errs,
        "avgLatencyMs": avg_lat,
        "avgExplainLatencyMs": avg_expl,
        "modelMeta": dict(_MODEL_META),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    with _metrics_lock:
        req_ok = int(_metrics["success_count"])
    uptime_s = float(time.perf_counter() - _SERVE_STARTED)
    return {
        "status": "ok",
        "modelLoaded": True,
        "shapReady": bool(_SHAP_WARMED),
        "uptime": _format_uptime_human(uptime_s),
        "requestCount": req_ok,
        "modelVersion": MODEL_VERSION,
        "explainVersion": EXPLAIN_VERSION,
        "features": str(len(_model_features)),
        "artifacts": {
            "model": "ok",
            "features": int(len(_model_features)),
            "backgroundRows": int(len(_SHAP_BACKGROUND)),
            "globalImportance": "yes" if _global_importance else "no",
        },
        "shap": "probability",
        "shap_background_rows": str(len(_SHAP_BACKGROUND)),
        "shap_background_ok": "yes" if _SHAP_BACKGROUND.shape[1] == len(_model_features) else "no",
        "global_importance": "yes" if _global_importance else "no",
        "shap_base_probability": f"{_SHAP_BASE_VALUE:.4f}",
        "shapDriftLogging": "true" if _SHAP_DRIFT_LOGGING else "false",
        "loggingLevel": str(_LOGGING_LEVEL_HINT),
        "shapWarmed": "true" if _SHAP_WARMED else "false",
        "explainBudgetMs": str(_EXPLAIN_BUDGET_MS),
        "explainMode": _explain_mode(),
        "trainingStatsLoaded": bool(_TRAINING_STATS),
    }
