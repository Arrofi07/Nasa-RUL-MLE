"""
🚀 NASA Turbofan RUL — Streamlit Dashboard

Tabs
----
1. Engine Explorer  — pick an engine, see its sensor degradation + XGBoost/LSTM RUL prediction
2. Batch Evaluation — run predictions on the full test set, show RMSE/MAE/R², scatter + error dist
3. Live Prediction  — manually enter a sensor reading and get an instant prediction

Run
---
    streamlit run app.py
    # API must already be running: python main.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
# API_URL = os.environ.get("API_URL", "https://nasa-rul-mle-production.up.railway.app")
PROCESSED_DIR = "data/processed"

# DROPPED_SENSORS = {"sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"}
# ALL_SENSORS = [f"sensor_{i}" for i in range(1, 22) if f"sensor_{i}" not in DROPPED_SENSORS]
ALL_SENSORS = [f"sensor_{i}" for i in range(1, 22)]

st.set_page_config(page_title="NASA Turbofan RUL", page_icon="✈️", layout="wide")

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data
def load_test_data():
    fe = pd.read_csv(f"{PROCESSED_DIR}/feature_engineered_test.csv")
    rul = pd.read_csv(f"{PROCESSED_DIR}/rul_clean.csv")
    rul.columns = ["rul"]

    # IMPORTANT: load the UNSCALED raw test file (output of load.py, before
    # preprocess.py applies StandardScaler). The API pipeline applies scaling
    # internally — sending test_clean.csv would double-scale and corrupt predictions.
    raw = pd.read_csv("data/raw/test.csv")

    return fe, rul, raw


def check_api() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def build_sensor_row(row: pd.Series) -> dict:
    d = {"engine_id": int(row["engine_id"]), "cycle": int(row["cycle"])}
    for col in ["setting_1", "setting_2", "setting_3"]:
        d[col] = float(row.get(col, 0.0))
    for i in range(1, 22):
        d[f"sensor_{i}"] = float(row.get(f"sensor_{i}", 0.0))
    return d


# How many cycles each model actually needs from history.
# Sending the full engine history (up to 300+ rows) wastes memory and
# serialisation time — the pipeline only uses the last TREE_WINDOW rows
# for rolling/diff features, and the last LSTM_SEQ rows for the sequence.
_TREE_WINDOW = 5   # rolling_window in InferencePipeline
_LSTM_SEQ = 41     # seq_len in InferencePipeline / lstm_config.json


def _trim(readings: list, n: int) -> list:
    """Return the last n readings (or all if fewer than n exist)."""
    return readings[-n:] if len(readings) > n else readings


def predict_xgb_batch(readings: list) -> float | None:
    try:
        r = requests.post(
            f"{API_URL}/predict/xgb/batch",
            json={"readings": _trim(readings, _TREE_WINDOW)},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["predicted_rul"]
    except Exception:
        return None


def predict_lgbm_batch(readings: list) -> float | None:
    try:
        r = requests.post(
            f"{API_URL}/predict/lgbm/batch",
            json={"readings": _trim(readings, _TREE_WINDOW)},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["predicted_rul"]
    except Exception:
        return None


def predict_lstm(readings: list) -> float | None:
    try:
        r = requests.post(
            f"{API_URL}/predict/lstm",
            json={"readings": _trim(readings, _LSTM_SEQ)},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["predicted_rul"]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("✈️ NASA Turbofan — Remaining Useful Life Dashboard")

api_ok = check_api()
if api_ok:
    st.success("API is online", icon="✅")
else:
    st.error("API is offline — start it with `python main.py` then refresh.", icon="🔴")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    ["🔍 Engine Explorer", "📊 Batch Evaluation", "🎛️ Live Prediction"]
)

# ===========================================================================
# TAB 1 — Engine Explorer
# ===========================================================================

with tab1:
    st.subheader("Engine Degradation Explorer")
    st.caption(
        "Select an engine from the test set, inspect its sensor signals over cycles, "
        "and compare XGBoost vs LSTM RUL predictions against the true RUL."
    )

    try:
        fe_df, rul_df, raw_df = load_test_data()
        engines = sorted(raw_df["engine_id"].unique())

        col_sel, col_sensor = st.columns([1, 2])
        with col_sel:
            engine_id = st.selectbox("Select Engine", engines)
            sensor_choice = st.selectbox(
                "Sensor to plot",
                ALL_SENSORS,
                index=ALL_SENSORS.index("sensor_2") if "sensor_2" in ALL_SENSORS else 0,
            )

        engine_raw = raw_df[raw_df["engine_id"] == engine_id].copy()
        engine_fe = fe_df[fe_df["engine_id"] == engine_id].copy()
        true_rul = float(rul_df.iloc[engine_id - 1]["rul"])
        n_cycles = len(engine_raw)

        with col_sensor:
            if sensor_choice in engine_raw.columns:
                fig = px.line(
                    engine_raw,
                    x="cycle",
                    y=sensor_choice,
                    title=f"Engine {engine_id} — {sensor_choice} over {n_cycles} cycles",
                    labels={"cycle": "Cycle", sensor_choice: "Normalised value"},
                )
                fig.update_traces(line_color="#1f77b4")
                st.plotly_chart(fig, width="stretch")
            else:
                st.info(f"{sensor_choice} was dropped during preprocessing.")

        st.markdown("---")
        st.markdown("#### RUL Predictions at end of recorded cycles")

        if st.button("Run Predictions", key="engine_pred"):
            if not api_ok:
                st.error("API is offline.")
            else:
                readings = [build_sensor_row(row) for _, row in engine_raw.iterrows()]

                with st.spinner("Calling XGBoost…"):
                    xgb_rul = predict_xgb_batch(readings)  # trims to last 5 internally
                with st.spinner("Calling LightGBM…"):
                    lgbm_rul = predict_lgbm_batch(readings)  # trims to last 5 internally
                with st.spinner("Calling LSTM…"):
                    lstm_rul = predict_lstm(readings)  # trims to last 41 internally

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("True RUL", f"{true_rul:.0f} cycles")
                c2.metric(
                    "XGBoost RUL",
                    f"{xgb_rul:.1f} cycles" if xgb_rul is not None else "—",
                    delta=f"{xgb_rul - true_rul:+.1f}" if xgb_rul is not None else None,
                )
                c3.metric(
                    "LightGBM RUL",
                    f"{lgbm_rul:.1f} cycles" if lgbm_rul is not None else "—",
                    delta=(
                        f"{lgbm_rul - true_rul:+.1f}" if lgbm_rul is not None else None
                    ),
                )
                c4.metric(
                    "LSTM RUL",
                    f"{lstm_rul:.1f} cycles" if lstm_rul is not None else "—",
                    delta=(
                        f"{lstm_rul - true_rul:+.1f}" if lstm_rul is not None else None
                    ),
                )

                results = {"Model": [], "Predicted RUL": []}
                results["Model"].append("True RUL")
                results["Predicted RUL"].append(true_rul)

                if xgb_rul is not None:
                    results["Model"].append("XGBoost")
                    results["Predicted RUL"].append(xgb_rul)

                if lgbm_rul is not None:
                    results["Model"].append("LightGBM")
                    results["Predicted RUL"].append(lgbm_rul)

                if lstm_rul is not None:
                    results["Model"].append("LSTM")
                    results["Predicted RUL"].append(lstm_rul)

                fig2 = px.bar(
                    pd.DataFrame(results),
                    x="Model",
                    y="Predicted RUL",
                    color="Model",
                    color_discrete_map={
                        "True RUL": "gray",
                        "XGBoost": "#1f77b4",
                        "LightGBM": "#2ca02c",
                        "LSTM": "#ff7f0e",
                    },
                    title=f"Engine {engine_id} — Predicted vs True RUL",
                )
                st.plotly_chart(fig2, width="stretch")

    except FileNotFoundError as e:
        st.error(f"Processed data not found: {e}\nRun the data pipeline first.")

# ===========================================================================
# TAB 2 — Batch Evaluation
# ===========================================================================

with tab2:
    st.subheader("Batch Evaluation on Test Set (100 engines)")
    st.caption(
        "Runs predictions for all 100 test engines and compares against ground-truth RUL."
    )

    model_choice = st.radio(
        "Model to evaluate", ["XGBoost", "LightGBM", "LSTM", "All"], horizontal=True
    )

    if st.button("Run Batch Evaluation", key="batch_eval"):
        if not api_ok:
            st.error("API is offline.")
        else:
            try:
                fe_df, rul_df, raw_df = load_test_data()
                engines = sorted(raw_df["engine_id"].unique())
                results = []
                progress = st.progress(0, text="Running predictions…")

                for i, eid in enumerate(engines):
                    engine_raw = raw_df[raw_df["engine_id"] == eid]
                    true_rul = float(rul_df.iloc[eid - 1]["rul"])
                    readings = [
                        build_sensor_row(row) for _, row in engine_raw.iterrows()
                    ]
                    row = {"engine_id": eid, "true_rul": true_rul}
                    if model_choice in ("XGBoost", "All"):
                        row["xgb_pred"] = predict_xgb_batch(readings)
                    if model_choice in ("LightGBM", "All"):
                        row["lgbm_pred"] = predict_lgbm_batch(readings)
                    if model_choice in ("LSTM", "All"):
                        row["lstm_pred"] = predict_lstm(readings)
                    results.append(row)
                    progress.progress(
                        (i + 1) / len(engines), text=f"Engine {eid}/{len(engines)}"
                    )

                progress.empty()
                results_df = pd.DataFrame(results)

                # Metrics
                st.markdown("#### Metrics")
                active_models = [
                    m
                    for m in ["XGBoost", "LightGBM", "LSTM"]
                    if model_choice == "All" or model_choice == m
                ]
                # 2 metric columns per model (RMSE + MAE)
                mcols = st.columns(len(active_models) * 2)

                def show_metrics(df, pred_col, label, cols, offset=0):
                    valid = df.dropna(subset=[pred_col])
                    err = valid[pred_col] - valid["true_rul"]
                    rmse = float(np.sqrt((err**2).mean()))
                    mae = float(err.abs().mean())
                    cols[offset].metric(f"{label} RMSE", f"{rmse:.2f}")
                    cols[offset + 1].metric(f"{label} MAE", f"{mae:.2f}")

                col_offset = 0
                if model_choice in ("XGBoost", "All"):
                    show_metrics(results_df, "xgb_pred", "XGBoost", mcols, col_offset)
                    col_offset += 2
                if model_choice in ("LightGBM", "All"):
                    show_metrics(results_df, "lgbm_pred", "LightGBM", mcols, col_offset)
                    col_offset += 2
                if model_choice in ("LSTM", "All"):
                    show_metrics(results_df, "lstm_pred", "LSTM", mcols, col_offset)

                # Scatter
                st.markdown("#### Predicted vs True RUL")
                fig = go.Figure()
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=0,
                    x1=125,
                    y1=150,
                    line=dict(color="gray", dash="dash"),
                )
                if model_choice in ("XGBoost", "All") and "xgb_pred" in results_df:
                    fig.add_trace(
                        go.Scatter(
                            x=results_df["true_rul"],
                            y=results_df["xgb_pred"],
                            mode="markers",
                            name="XGBoost",
                            marker=dict(color="#1f77b4", size=7, opacity=0.7),
                        )
                    )
                if model_choice in ("LightGBM", "All") and "lgbm_pred" in results_df:
                    fig.add_trace(
                        go.Scatter(
                            x=results_df["true_rul"],
                            y=results_df["lgbm_pred"],
                            mode="markers",
                            name="LightGBM",
                            marker=dict(color="#2ca02c", size=7, opacity=0.7),
                        )
                    )
                if model_choice in ("LSTM", "All") and "lstm_pred" in results_df:
                    fig.add_trace(
                        go.Scatter(
                            x=results_df["true_rul"],
                            y=results_df["lstm_pred"],
                            mode="markers",
                            name="LSTM",
                            marker=dict(color="#ff7f0e", size=7, opacity=0.7),
                        )
                    )
                fig.update_layout(
                    xaxis_title="True RUL (cycles)",
                    yaxis_title="Predicted RUL (cycles)",
                    title="Predicted vs True RUL — Test Set",
                )
                st.plotly_chart(fig, width="stretch")

                # Error distribution
                st.markdown("#### Prediction Error Distribution")
                err_data = []
                if model_choice in ("XGBoost", "All") and "xgb_pred" in results_df:
                    err_data.append(
                        go.Histogram(
                            x=(
                                results_df["xgb_pred"] - results_df["true_rul"]
                            ).dropna(),
                            name="XGBoost",
                            opacity=0.7,
                            marker_color="#1f77b4",
                        )
                    )
                if model_choice in ("LightGBM", "All") and "lgbm_pred" in results_df:
                    err_data.append(
                        go.Histogram(
                            x=(
                                results_df["lgbm_pred"] - results_df["true_rul"]
                            ).dropna(),
                            name="LightGBM",
                            opacity=0.7,
                            marker_color="#2ca02c",
                        )
                    )
                if model_choice in ("LSTM", "All") and "lstm_pred" in results_df:
                    err_data.append(
                        go.Histogram(
                            x=(
                                results_df["lstm_pred"] - results_df["true_rul"]
                            ).dropna(),
                            name="LSTM",
                            opacity=0.7,
                            marker_color="#ff7f0e",
                        )
                    )
                fig_err = go.Figure(err_data)
                fig_err.update_layout(
                    barmode="overlay",
                    xaxis_title="Error (cycles)",
                    yaxis_title="Count",
                    title="Error Distribution (Predicted − True)",
                )
                st.plotly_chart(fig_err, width="stretch")

                with st.expander("View full results table"):
                    st.dataframe(results_df.round(2), width="stretch")

            except FileNotFoundError as e:
                st.error(f"Processed data not found: {e}")

# ===========================================================================
# TAB 3 — Live Prediction
# ===========================================================================

with tab3:
    st.subheader("Live Single-Cycle Prediction")
    st.caption(
        "Enter raw sensor values manually and get an instant XGBoost RUL prediction."
    )

    s_col1, s_col2 = st.columns(2)
    engine_id_in = s_col1.number_input("Engine ID", min_value=1, max_value=999, value=1)
    cycle_in = s_col2.number_input("Cycle", min_value=1, max_value=500, value=50)

    st.markdown("#### Settings")
    c1, c2, c3 = st.columns(3)
    setting_1 = c1.number_input("Setting 1", value=-0.0007, format="%.4f")
    setting_2 = c2.number_input("Setting 2", value=-0.0004, format="%.4f")
    setting_3 = c3.number_input("Setting 3", value=100.0, format="%.2f")

    st.markdown("#### Sensor Values")
    defaults = {
        "sensor_1": 518.67,
        "sensor_2": 641.82,
        "sensor_3": 1589.70,
        "sensor_4": 1400.60,
        "sensor_5": 14.62,
        "sensor_6": 21.61,
        "sensor_7": 554.36,
        "sensor_8": 2388.02,
        "sensor_9": 9046.19,
        "sensor_10": 1.30,
        "sensor_11": 47.47,
        "sensor_12": 521.66,
        "sensor_13": 2388.02,
        "sensor_14": 8138.62,
        "sensor_15": 8.4195,
        "sensor_16": 0.03,
        "sensor_17": 392.0,
        "sensor_18": 2388.0,
        "sensor_19": 100.0,
        "sensor_20": 39.06,
        "sensor_21": 23.419,
    }
    sensor_values = {}
    sensor_list = [f"sensor_{i}" for i in range(1, 22)]
    for row_sensors in [sensor_list[i : i + 7] for i in range(0, 21, 7)]:
        cols = st.columns(len(row_sensors))
        for col, sname in zip(cols, row_sensors):
            sensor_values[sname] = col.number_input(
                sname, value=defaults[sname], format="%.4f", key=f"live_{sname}"
            )

    if st.button("Predict RUL", key="live_pred"):
        if not api_ok:
            st.error("API is offline.")
        else:
            payload = {
                "engine_id": int(engine_id_in),
                "cycle": int(cycle_in),
                "setting_1": setting_1,
                "setting_2": setting_2,
                "setting_3": setting_3,
                **sensor_values,
            }
            with st.spinner("Getting prediction…"):
                try:
                    r = requests.post(
                        f"{API_URL}/predict/xgb", json=payload, timeout=10
                    )
                    r.raise_for_status()
                    result = r.json()
                    st.success("Prediction complete!")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Engine", result["engine_id"])
                    m2.metric("Cycle", result["cycle"])
                    m3.metric("Predicted RUL", f"{result['predicted_rul']} cycles")

                    rul_val = result["predicted_rul"]
                    color = (
                        "#2ecc71"
                        if rul_val > 80
                        else "#f39c12" if rul_val > 30 else "#e74c3c"
                    )

                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=rul_val,
                            title={"text": "Remaining Useful Life (cycles)"},
                            gauge={
                                "axis": {"range": [0, 125]},
                                "bar": {"color": color},
                                "steps": [
                                    {"range": [0, 30], "color": "#fdecea"},
                                    {"range": [30, 80], "color": "#fef9e7"},
                                    {"range": [80, 125], "color": "#eafaf1"},
                                ],
                                "threshold": {
                                    "line": {"color": "red", "width": 3},
                                    "thickness": 0.75,
                                    "value": 30,
                                },
                            },
                        )
                    )
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, width="stretch")

                except requests.HTTPError:
                    st.error(f"API error {r.status_code}: {r.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")