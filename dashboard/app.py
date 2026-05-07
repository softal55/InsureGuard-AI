"""InsureGuard AI — Streamlit dashboard for the fraud ML API."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="InsureGuard AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

HEALTH_URL = os.environ.get("INSUREGUARD_HEALTH_URL", "http://127.0.0.1:8000/health")
API_URL = os.environ.get("INSUREGUARD_API_URL", "http://127.0.0.1:8000/predict")
REQUEST_TIMEOUT_S = float(os.environ.get("INSUREGUARD_REQUEST_TIMEOUT_S", "120"))

PLOT_TEMPLATE = "plotly_dark"
PLOT_FONT = dict(family="Inter, system-ui, sans-serif", color="#e2e8f0")


def _init_session() -> None:
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "health_snapshot" not in st.session_state:
        st.session_state.health_snapshot = None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _call_predict(payload: dict, explain: str | None) -> tuple[dict | None, str | None]:
    params: dict[str, str] = {}
    if explain is not None:
        params["explain"] = explain
    try:
        r = requests.post(
            API_URL,
            params=params,
            json=payload,
            timeout=REQUEST_TIMEOUT_S,
        )
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach API at `{API_URL}`. Start the FastAPI app (`uvicorn main:app` in `ml-service-python`)."
    except requests.exceptions.Timeout:
        return None, "Request timed out (first SHAP call can take over a minute)."
    except requests.exceptions.RequestException as e:
        return None, str(e)

    if r.status_code != 200:
        try:
            detail = r.json()
        except Exception:
            detail = r.text[:500]
        return None, f"HTTP {r.status_code}: {detail}"

    try:
        return r.json(), None
    except ValueError:
        return None, "Invalid JSON from API."


def _fetch_health() -> dict[str, Any] | None:
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def _explain_banner(result: dict) -> str:
    status = str(result.get("explanationStatus", ""))
    if status == "ok":
        return ""
    if status == "disabled":
        reason = result.get("explanationSkippedReason")
        if reason == "budget_exceeded":
            return (
                "Explanation skipped (pre-explain budget exceeded). "
                "Raise `INSUREGUARD_EXPLAIN_BUDGET_MS` on the server if needed."
            )
        if reason == "explain_disabled":
            return "Explanation disabled (`INSUREGUARD_EXPLAIN_OFF`)."
        return "Explanation unavailable."
    if status == "fallback":
        err = result.get("explanationError") or "unknown"
        return f"Partial explanation (`{err}`). Reconstruction values may still be present."
    return ""


def _risk_badge_class(level: str) -> str:
    low = str(level).lower()
    if low == "high":
        return "rb-high"
    if low == "low":
        return "rb-low"
    return "rb-mid"


# -----------------------------------------------------------------------------
# Styling (pro dashboard)
# -----------------------------------------------------------------------------
st.markdown(
    f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
  html, body, [class*="css"] {{
    font-family: "DM Sans", system-ui, sans-serif;
  }}
  .stApp {{
    background: linear-gradient(165deg, #0b1220 0%, #020617 45%, #0f172a 100%);
    color: #e2e8f0;
  }}
  section[data-testid="stSidebar"] > div {{
    background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
    border-right: 1px solid #1e293b;
  }}
  .ig-hero {{
    padding: 0.25rem 0 1.25rem 0;
    border-bottom: 1px solid rgba(30, 41, 59, 0.8);
    margin-bottom: 1rem;
  }}
  .ig-hero h1 {{
    font-size: clamp(1.5rem, 2.5vw, 2rem);
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
    background: linear-gradient(90deg, #38bdf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  .ig-hero p {{
    margin: 0.35rem 0 0 0;
    color: #94a3b8;
    font-size: 0.95rem;
  }}
  .ig-panel {{
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(51, 65, 85, 0.6);
    border-radius: 16px;
    padding: 1.25rem 1.35rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
  }}
  .ig-panel-title {{
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 0.75rem;
  }}
  div[data-testid="stMetric"] {{
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 0.65rem 0.5rem;
  }}
  div[data-testid="stMetric"] label {{
    color: #94a3b8 !important;
  }}
  div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: #f1f5f9 !important;
  }}
  .rb-high {{ display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; background: rgba(239,68,68,0.15); color: #fca5a5; font-weight: 600; font-size: 0.85rem; }}
  .rb-low {{ display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; background: rgba(34,197,94,0.15); color: #86efac; font-weight: 600; font-size: 0.85rem; }}
  .rb-mid {{ display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; background: rgba(148,163,184,0.12); color: #cbd5e1; font-size: 0.85rem; }}
  #MainMenu {{visibility: hidden;}}
  footer {{visibility: hidden;}}
</style>
""",
    unsafe_allow_html=True,
)


_init_session()

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ InsureGuard")
    st.caption("Fraud risk · SHAP explanations")

    explain_mode = st.selectbox(
        "Explanation depth",
        ["compact (top 3)", "full (top 10)", "default (top 5)"],
        index=0,
        help="Maps to the API `explain` query: compact, full, or omitted for server default.",
    )
    if explain_mode.startswith("compact"):
        explain_query: str | None = "compact"
    elif explain_mode.startswith("full"):
        explain_query = "full"
    else:
        explain_query = None

    if st.button("↻ Check API health", use_container_width=True):
        st.session_state.health_snapshot = _fetch_health()

    hs = st.session_state.health_snapshot
    if hs is not None:
        st.success(f"API **{hs.get('status', '?')}** · `{hs.get('explainMode', '?')}` mode")
        st.caption(f"Model: `{hs.get('modelVersion', '')}`")
    else:
        st.caption("Health not checked yet. Use the button above.")

    st.divider()
    st.markdown("**Endpoints**")
    st.code(f"POST {API_URL}", language="text")
    st.caption("Override with `INSUREGUARD_API_URL` / `INSUREGUARD_HEALTH_URL`.")

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class="ig-hero">
  <h1>Claim intelligence</h1>
  <p>Score claims against the trained XGBoost model with SHAP-based explanations.</p>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Input row
# -----------------------------------------------------------------------------
with st.container():
    st.markdown('<p class="ig-panel-title">Claim input</p>', unsafe_allow_html=True)
    total_claim_amount = st.number_input("Total claim amount", 0, 1_000_000, 25000, step=100)
    vehicle_claim = st.number_input("Vehicle claim", 0, 500_000, 8000, step=100)
    property_claim = st.number_input("Property claim", 0, 500_000, 7000, step=100)
    incident_hour = st.slider("Incident hour (0–23)", 0, 23, 14)

    analyze = st.button("Run analysis", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Analyze
# -----------------------------------------------------------------------------
if analyze:
    payload = {
        "total_claim_amount": total_claim_amount,
        "vehicle_claim": vehicle_claim,
        "property_claim": property_claim,
        "incident_hour_of_day": incident_hour,
    }
    with st.spinner("Calling ML service…"):
        result, err = _call_predict(payload, explain_query)
    if err:
        st.session_state.last_error = err
        st.session_state.last_result = None
    else:
        st.session_state.last_error = None
        st.session_state.last_result = result

# Errors (full width, always visible)
if st.session_state.last_error:
    st.error(st.session_state.last_error)

# -----------------------------------------------------------------------------
# Results (full width — better for charts)
# -----------------------------------------------------------------------------
result = st.session_state.last_result
if result:
    risk_score = _safe_float(result.get("riskScore"))
    fraud_prob = _safe_float(result.get("fraudProbability", result.get("riskScore")))
    band = str(result.get("confidenceBand", "—"))
    latency = result.get("latency") or {}
    total_ms = int(latency.get("totalMs", 0))
    explain_ms = int(latency.get("explainMs", result.get("explainCostMs", 0)))
    risk_level = str(result.get("riskLevel", ""))

    st.markdown('<div class="ig-panel-title" style="margin-top:1rem">Results</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Risk score", f"{risk_score:.1%}")
    m2.metric("Fraud probability", f"{fraud_prob:.1%}")
    m3.metric("Confidence band", band.replace("_", " "))
    m4.metric("Total latency", f"{total_ms} ms")
    m5.metric("Explain time", f"{explain_ms} ms")

    badge_cls = _risk_badge_class(risk_level)
    ready = bool(result.get("explanationReady", False))
    st.markdown(
        f'<p style="margin:0.5rem 0 0 0;"><span class="{badge_cls}">{risk_level or "—"}</span>'
        f' &nbsp;·&nbsp; Explanation ready: <b>{"Yes" if ready else "No"}</b>'
        f' &nbsp;·&nbsp; <code style="color:#64748b">{result.get("requestId", "")}</code></p>',
        unsafe_allow_html=True,
    )

    ci = result.get("confidenceInterval")
    if isinstance(ci, list) and len(ci) == 2:
        st.caption(
            f"Demo interval ({result.get('confidenceIntervalType', '')}): "
            f"[{float(ci[0]):.4f}, {float(ci[1]):.4f}] · ε = {result.get('confidenceEpsUsed', '—')}"
        )

    g_left, g_right = st.columns([1.1, 1])
    with g_left:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=fraud_prob * 100,
                number={"suffix": " %", "valueformat": ".1f"},
                title={"text": "Fraud probability", "font": PLOT_FONT},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569"},
                    "bar": {"color": "#22c55e" if fraud_prob < 0.5 else "#ef4444"},
                    "bgcolor": "rgba(15,23,42,0.9)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50], "color": "rgba(34,197,94,0.12)"},
                        {"range": [50, 100], "color": "rgba(239,68,68,0.12)"},
                    ],
                    "threshold": {
                        "line": {"color": "#f8fafc", "width": 2},
                        "thickness": 0.85,
                        "value": 50,
                    },
                },
            )
        )
        gauge.update_layout(
            template=PLOT_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=PLOT_FONT,
            height=300,
            margin=dict(t=48, b=24, l=24, r=24),
        )
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

    with g_right:
        share = result.get("latencyShare") or {}
        sm = float(share.get("model", 0) or 0)
        se = float(share.get("explain", 0) or 0)
        if total_ms > 0 and (sm + se) > 1e-9:
            pie = go.Figure(
                data=[
                    go.Pie(
                        labels=["Model + encode", "Explain + post"],
                        values=[sm, se],
                        hole=0.58,
                        marker=dict(colors=["#38bdf8", "#a78bfa"], line=dict(color="#0f172a", width=2)),
                        textinfo="label+percent",
                        textfont=dict(size=12, color="#e2e8f0"),
                    )
                ]
            )
            pie.update_layout(
                template=PLOT_TEMPLATE,
                paper_bgcolor="rgba(0,0,0,0)",
                font=PLOT_FONT,
                showlegend=False,
                height=300,
                margin=dict(t=40, b=20, l=20, r=20),
                title=dict(text="Latency share", x=0.5, font=dict(size=14, color="#94a3b8")),
            )
            st.plotly_chart(pie, use_container_width=True, config={"displayModeBar": False})
        else:
            st.empty()

    warnings = result.get("warnings") or []
    if warnings:
        for w in warnings:
            st.warning(w)

    st.markdown("### Explainability")

    ex_status = str(result.get("explanationStatus", ""))
    banner = _explain_banner(result)
    if banner:
        st.info(banner)

    if ex_status == "ok" and result.get("topFactors"):
        df = pd.DataFrame(result["topFactors"])
        if not df.empty and "impact" in df.columns and "feature" in df.columns:
            tab1, tab2, tab3 = st.tabs(["Factor impact", "Waterfall", "Table"])
            with tab1:
                fig = px.bar(
                    df,
                    x="impact",
                    y="feature",
                    orientation="h",
                    color="direction",
                    color_discrete_map={
                        "increase_risk": "#f87171",
                        "decrease_risk": "#4ade80",
                        "neutral": "#94a3b8",
                    },
                )
                fig.update_layout(
                    template=PLOT_TEMPLATE,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.4)",
                    font=PLOT_FONT,
                    height=max(360, 28 * len(df)),
                    margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(categoryorder="total ascending"),
                    xaxis_title="Impact",
                    yaxis_title="",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                wf = go.Figure(
                    go.Waterfall(
                        name="SHAP",
                        orientation="v",
                        measure=["relative"] * len(df),
                        x=df["feature"].astype(str).tolist(),
                        y=df["impact"].astype(float).tolist(),
                        connector={"line": {"color": "#475569", "width": 1}},
                    )
                )
                wf.update_layout(
                    template=PLOT_TEMPLATE,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.4)",
                    font=PLOT_FONT,
                    height=max(420, 36 * len(df)),
                    margin=dict(l=8, r=8, t=8, b=120),
                    xaxis=dict(tickangle=-40),
                    showlegend=False,
                )
                st.plotly_chart(wf, use_container_width=True)

            with tab3:
                cols_show = [c for c in ("rank", "feature", "impact", "direction", "percent", "group") if c in df.columns]
                st.dataframe(
                    df[cols_show],
                    use_container_width=True,
                    hide_index=True,
                )

            st.subheader("Model reconstruction")
            r1, r2, r3, r4 = st.columns(4)
            if "baseValue" in result:
                r1.metric("Base (SHAP)", f"{float(result['baseValue']):.4f}")
            if "approxProbability" in result:
                r2.metric("Approx. probability", f"{float(result['approxProbability']):.4f}")
            if result.get("approxDelta") is not None:
                r3.metric("Approx. delta", f"{float(result['approxDelta']):.6f}")
            if result.get("approxMatches") is not None:
                r4.metric("Matches model prob.", "Yes" if result["approxMatches"] else "No")

    elif ex_status == "fallback" and result.get("approxProbability") is not None:
        st.warning("Top factors unavailable; SHAP reconstruction may still be shown below.")
        c1, c2, c3 = st.columns(3)
        if "baseValue" in result:
            c1.metric("Base value", f"{float(result['baseValue']):.4f}")
        if "approxProbability" in result:
            c2.metric("Approx. probability", f"{float(result['approxProbability']):.4f}")
        if "directionScore" in result:
            c3.metric("Direction score", f"{float(result['directionScore']):.4f}")

    else:
        st.caption(
            f"Status `{ex_status}` · ready reason: `{result.get('explanationReadyReason', '')}`"
        )

    with st.expander("System metadata"):
        meta = {
            k: result.get(k)
            for k in (
                "modelVersion",
                "contractVersion",
                "payloadVersion",
                "modelFamily",
                "modelSignature",
                "modelSignatureParts",
                "schemaHash",
                "schemaCanonVersion",
                "explainMode",
                "explainBudgetMs",
                "explainPolicy",
                "featuresUsed",
                "backgroundSource",
            )
        }
        st.json({k: v for k, v in meta.items() if v is not None})

    with st.expander("Raw API response"):
        st.json(result)

else:
    st.markdown(
        """
<div class="ig-panel" style="margin-top:0.5rem">
  <p style="color:#94a3b8;margin:0;">Adjust claim fields, then click <b>Run analysis</b>.
  The last successful response stays visible until you run again (or the API errors).</p>
</div>
""",
        unsafe_allow_html=True,
    )
