"""
🚀 NASA Turbofan RUL — Streamlit Dashboard

Tabs
----
1. Engine Explorer  — pick an engine, see its sensor degradation + XGBoost/LSTM RUL prediction
2. Batch Evaluation — run predictions on the full test set, show RMSE/MAE/R², scatter + error dist
3. Live Prediction  — manually enter a sensor reading and get an instant prediction
4. Model Comparison - 

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


def predict_xgb_batch(readings: list) -> float | None:
    try:
        r = requests.post(
            f"{API_URL}/predict/xgb/batch", json={"readings": readings}, timeout=10
        )
        r.raise_for_status()
        return r.json()["predicted_rul"]
    except Exception:
        return None


def predict_lgbm_batch(readings: list) -> float | None:
    try:
        r = requests.post(
            f"{API_URL}/predict/lgbm/batch", json={"readings": readings}, timeout=10
        )
        r.raise_for_status()
        return r.json()["predicted_rul"]
    except Exception:
        return None


def predict_lstm(readings: list) -> float | None:
    try:
        r = requests.post(
            f"{API_URL}/predict/lstm", json={"readings": readings}, timeout=10
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

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Engine Explorer", "📊 Batch Evaluation", "🎛️ Live Prediction", "🏆 Model Comparison"]
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
                    xgb_rul = predict_xgb_batch(readings)
                with st.spinner("Calling LightGBM…"):
                    lgbm_rul = predict_lgbm_batch(readings)
                with st.spinner("Calling LSTM…"):
                    lstm_rul = predict_lstm(readings)

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

# ===========================================================================
# TAB 4 — Model Comparison
# ===========================================================================
# This tab answers one question: 
# "when should I trust XGBoost or LightGBM over LSTM (or vice versa)?"
# It does this in three steps:
#   1. Load the batch results that were already computed in Batch Evaluation (or
#      offer to run them now if the user hasn't yet).
#   2. Slice the results by RUL range (early / mid / late life) and engine
#      lifetime length (short / medium / long) to show where each model wins.
#   3. Display residual plots, per-segment bar charts, and a head-to-head
#      scatter so the user can build intuition about each model's strengths.

with tab4:
    st.subheader("Model Comparison — When Does Each Model Win?")
    st.caption(
        "Run Batch Evaluation first (All models), then come here to see "
        "where XGBoost, LightGBM, and LSTM each outperform the others."
    )

    # ---------------------------------------------------------------------------
    # Colour palette — consistent across every chart in this tab
    # ---------------------------------------------------------------------------
    MODEL_COLORS = {
        "XGBoost":  "#1f77b4",   # blue
        "LightGBM": "#2ca02c",   # green
        "LSTM":     "#ff7f0e",   # orange
    }

    try:
        # -----------------------------------------------------------------------
        # 1. DATA — load the test set and attempt to get batch predictions
        # -----------------------------------------------------------------------

        fe_df, rul_df, raw_df = load_test_data()

        # Count how many cycles each test engine has in the raw file.
        # We use this later to label engines as "short / medium / long lived".
        engine_lengths = (
            raw_df.groupby("engine_id")["cycle"]
            .max()
            .rename("engine_length")
            .reset_index()
        )

        st.markdown("### Step 1 — Run predictions for all engines")
        st.info(
            "This section needs predictions from **all three models** across all "
            "100 test engines. Click the button below. It calls the live API "
            "for every engine (~30 s depending on connection speed).",
            icon="ℹ️",
        )

        # We store the results in Streamlit session state so the user doesn't
        # have to re-run every time they switch tabs.
        # session_state persists for the lifetime of the browser session.
        if "comparison_df" not in st.session_state:
            st.session_state["comparison_df"] = None

        if st.button("Run All-Model Batch Predictions", key="compare_run"):
            if not api_ok:
                st.error("API is offline — start it with `python main.py` then refresh.")
            else:
                engines = sorted(raw_df["engine_id"].unique())
                results = []
                bar = st.progress(0, text="Gathering predictions…")

                for i, eid in enumerate(engines):
                    engine_raw = raw_df[raw_df["engine_id"] == eid]
                    true_rul   = float(rul_df.iloc[eid - 1]["rul"])

                    # Build the list of sensor dicts for this engine.
                    # The helper trims to the last N cycles inside each predict_* call.
                    readings = [
                        build_sensor_row(row) for _, row in engine_raw.iterrows()
                    ]

                    results.append({
                        "engine_id":  eid,
                        "true_rul":   true_rul,
                        "xgb_pred":   predict_xgb_batch(readings),
                        "lgbm_pred":  predict_lgbm_batch(readings),
                        "lstm_pred":  predict_lstm(readings),
                    })

                    bar.progress((i + 1) / len(engines), text=f"Engine {eid}/{len(engines)}")

                bar.empty()

                df = pd.DataFrame(results)

                # Compute absolute errors for each model.
                # Absolute error = |predicted − true|.  Lower is better.
                df["xgb_err"]  = (df["xgb_pred"]  - df["true_rul"]).abs()
                df["lgbm_err"] = (df["lgbm_pred"] - df["true_rul"]).abs()
                df["lstm_err"] = (df["lstm_pred"] - df["true_rul"]).abs()

                # Compute signed errors (predicted − true).
                # Positive = over-predicts (thinks engine has more life than it does).
                # Negative = under-predicts (too pessimistic about remaining life).
                df["xgb_resid"]  = df["xgb_pred"]  - df["true_rul"]
                df["lgbm_resid"] = df["lgbm_pred"] - df["true_rul"]
                df["lstm_resid"] = df["lstm_pred"] - df["true_rul"]

                # Label which model has the smallest absolute error for each engine.
                # This is used later to show "win rate" per segment.
                def best_model(row):
                    errors = {
                        "XGBoost":  row["xgb_err"],
                        "LightGBM": row["lgbm_err"],
                        "LSTM":     row["lstm_err"],
                    }
                    # Drop any model that returned None (API failure)
                    errors = {k: v for k, v in errors.items() if pd.notna(v)}
                    return min(errors, key=errors.get) if errors else None

                df["best_model"] = df.apply(best_model, axis=1)

                # Merge in the engine lifetime length we computed earlier
                df = df.merge(engine_lengths, on="engine_id")

                # Save to session state so it survives tab switches
                st.session_state["comparison_df"] = df
                st.success("Done! Scroll down to explore the comparison.")

        # -----------------------------------------------------------------------
        # 2. ANALYSIS — only shown once we have results
        # -----------------------------------------------------------------------

        df = st.session_state["comparison_df"]

        if df is not None:

            # -------------------------------------------------------------------
            # Section 2a — Overall metric summary
            # -------------------------------------------------------------------
            st.markdown("---")
            st.markdown("### Overall Test-Set Metrics")
            st.caption(
                "RMSE penalises large errors more than MAE does. "
                "A model with low MAE but high RMSE is making occasional very bad predictions."
            )

            m_cols = st.columns(3)
            for col, (model, pred_col) in zip(
                m_cols,
                [("XGBoost", "xgb_pred"), ("LightGBM", "lgbm_pred"), ("LSTM", "lstm_pred")],
            ):
                valid = df.dropna(subset=[pred_col])
                err   = valid[pred_col] - valid["true_rul"]
                rmse  = float(np.sqrt((err ** 2).mean()))
                mae   = float(err.abs().mean())
                r2    = float(
                    1 - (
                        (err ** 2).sum() / 
                        ((valid["true_rul"] - valid["true_rul"].mean()) ** 2).sum()
                        )
                )
                with col:
                    st.markdown(
                        f"<div style='text-align:center; padding:12px; border-radius:8px; "
                        f"background:{MODEL_COLORS[model]}22"
                        f"border:1px solid {MODEL_COLORS[model]}'>"
                        f"<b style='color:{MODEL_COLORS[model]}'>{model}</b><br>"
                        f"RMSE <b>{rmse:.2f}</b> &nbsp;|&nbsp"
                        f"MAE <b>{mae:.2f}</b> &nbsp;|&nbsp"
                        f"R² <b>{r2:.3f}</b>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # -------------------------------------------------------------------
            # Section 2b — Win-rate pie chart
            # Each engine is assigned to whichever model had the lowest absolute
            # error. The pie chart shows how often each model "won".
            # -------------------------------------------------------------------
            st.markdown("---")
            st.markdown("### Who wins most often?")
            st.caption(
                "Each engine is assigned to the model that predicted its RUL "
                "most accurately. The chart shows how many engines each model "
                "won across the full 100-engine test set."
            )

            win_counts = df["best_model"].value_counts().reset_index()
            win_counts.columns = ["Model", "Engines Won"]

            fig_pie = px.pie(
                win_counts,
                names="Model",
                values="Engines Won",
                color="Model",
                color_discrete_map=MODEL_COLORS,
                title="Win Rate — Best Model per Engine (lowest absolute error)",
            )
            fig_pie.update_traces(textinfo="label+percent+value")
            st.plotly_chart(fig_pie, use_container_width=True)

            # -------------------------------------------------------------------
            # Section 2c — Performance by RUL range
            # -------------------------------------------------------------------
            # We split the test engines into three life-stage buckets:
            #   Early  (true RUL > 80): engine still has a long time left
            #   Mid    (30 < true RUL ≤ 80): engine is in the middle of life
            #   Late   (true RUL ≤ 30): engine is near end of life
            #
            # This matters for maintenance: a model that is only accurate when
            # the engine is nearly dead is not useful — you want warning early.
            # -------------------------------------------------------------------
            st.markdown("---")
            st.markdown("### Performance by RUL range (life stage)")
            st.caption(
                "Early life = RUL > 80 cycles, Mid life = 30–80 cycles, "
                "Late life = RUL ≤ 30 cycles. "
                "A good model should be accurate across all three stages, "
                "but accuracy near end-of-life is most critical for maintenance."
            )

            # Assign each engine to a life-stage bucket
            def rul_bucket(rul):
                if rul > 80:
                    return "Early (RUL > 80)"
                elif rul > 30:
                    return "Mid (30 < RUL ≤ 80)"
                else:
                    return "Late (RUL ≤ 30)"

            df["rul_stage"] = df["true_rul"].apply(rul_bucket)

            # Build a tidy long-form dataframe: one row per (engine, model)
            # with columns [rul_stage, model, mae].
            # This format is easiest to plot with plotly express bar charts.
            stage_rows = []
            for _, row in df.iterrows():
                for model, col in [("XGBoost", "xgb_err"), 
                                   ("LightGBM", "lgbm_err"), 
                                   ("LSTM", "lstm_err")]:
                    if pd.notna(row[col]):
                        stage_rows.append({
                            "rul_stage": row["rul_stage"],
                            "model":     model,
                            "abs_error": row[col],
                        })

            stage_df = pd.DataFrame(stage_rows)

            # Group by stage + model and compute mean absolute error
            stage_summary = (
                stage_df.groupby(["rul_stage", "model"])["abs_error"]
                .mean()
                .reset_index()
                .rename(columns={"abs_error": "Mean Absolute Error"})
            )

            # Sort stages in a logical order (early → mid → late)
            stage_order = ["Early (RUL > 80)", "Mid (30 < RUL ≤ 80)", "Late (RUL ≤ 30)"]
            stage_summary["rul_stage"] = pd.Categorical(
                stage_summary["rul_stage"], categories=stage_order, ordered=True
            )
            stage_summary = stage_summary.sort_values("rul_stage")

            fig_stage = px.bar(
                stage_summary,
                x="rul_stage",
                y="Mean Absolute Error",
                color="model",
                barmode="group",
                color_discrete_map=MODEL_COLORS,
                title="Mean Absolute Error by Life Stage",
                labels={"rul_stage": "Life Stage", "model": "Model"},
                text_auto=".1f",   # show the MAE value on each bar
            )
            fig_stage.update_traces(textposition="outside")
            st.plotly_chart(fig_stage, use_container_width=True)

            # Add a written interpretation so the viewer doesn't have to guess
            # which model is better at each stage
            with st.expander("📖 How to read this chart"):
                st.markdown(
                    """
                    - **Lower bar = better** (lower mean absolute error).
                    - **Early life** engines have a lot of cycles left (RUL > 80).
                      Tree models often struggle here because the sensor signals
                      are still near-healthy and there is less pattern to learn.
                    - **Late life** engines are close to failure (RUL ≤ 30).
                      This is the most important zone — a wrong prediction here
                      means either unnecessary early maintenance or dangerous overrun.
                    - **LSTM** tends to perform better in early and mid life because
                      it sees the full temporal sequence and can detect subtle trends.
                    - **XGBoost / LightGBM** use only the last 5 cycles (rolling
                      window) so they react faster to sudden changes, which can
                      help or hurt in late life depending on the engine.
                    """
                )

            # -------------------------------------------------------------------
            # Section 2d — Performance by engine lifetime length
            # -------------------------------------------------------------------
            # Engines have very different total lifetimes (128–362 cycles).
            # Short-lived engines give the model fewer cycles to learn the
            # degradation pattern, which may affect accuracy.
            # -------------------------------------------------------------------
            st.markdown("---")
            st.markdown("### Performance by engine lifetime length")
            st.caption(
                "Short-lived engines (< 175 cycles) give models fewer data points "
                "to work with. Long-lived engines (≥ 250 cycles) have richer history "
                "but the degradation signal may be slower and harder to detect early."
            )

            def length_bucket(n):
                if n < 175:
                    return "Short (< 175 cycles)"
                elif n < 250:
                    return "Medium (175–249 cycles)"
                else:
                    return "Long (≥ 250 cycles)"

            df["lifetime_group"] = df["engine_length"].apply(length_bucket)

            length_rows = []
            for _, row in df.iterrows():
                for model, col in [("XGBoost", "xgb_err"), 
                                   ("LightGBM", "lgbm_err"), 
                                   ("LSTM", "lstm_err")]:
                    if pd.notna(row[col]):
                        length_rows.append({
                            "lifetime_group": row["lifetime_group"],
                            "model":          model,
                            "abs_error":      row[col],
                        })

            length_df = pd.DataFrame(length_rows)

            length_summary = (
                length_df.groupby(["lifetime_group", "model"])["abs_error"]
                .mean()
                .reset_index()
                .rename(columns={"abs_error": "Mean Absolute Error"})
            )

            length_order = ["Short (< 175 cycles)", 
                            "Medium (175–249 cycles)", 
                            "Long (≥ 250 cycles)"]
            length_summary["lifetime_group"] = pd.Categorical(
                length_summary["lifetime_group"], categories=length_order, ordered=True
            )
            length_summary = length_summary.sort_values("lifetime_group")

            fig_len = px.bar(
                length_summary,
                x="lifetime_group",
                y="Mean Absolute Error",
                color="model",
                barmode="group",
                color_discrete_map=MODEL_COLORS,
                title="Mean Absolute Error by Engine Lifetime Length",
                labels={"lifetime_group": "Engine Lifetime", "model": "Model"},
                text_auto=".1f",
            )
            fig_len.update_traces(textposition="outside")
            st.plotly_chart(fig_len, use_container_width=True)

            # -------------------------------------------------------------------
            # Section 2e — Residual plots (signed error vs true RUL)
            # -------------------------------------------------------------------
            # A residual plot shows signed error (predicted − true) on the Y axis
            # and true RUL on the X axis.
            # What to look for:
            #   • Points scattered evenly around 0 → good (no systematic bias)
            #   • Points mostly above 0 → model over-predicts (too optimistic)
            #   • Points mostly below 0 → model under-predicts (too pessimistic)
            #   • A slope pattern → model is biased for long/short RUL values
            # -------------------------------------------------------------------
            st.markdown("---")
            st.markdown("### Residual plots (signed prediction error)")
            st.caption(
                "Residual = Predicted − True RUL. "
                "Points above 0 mean the model over-estimated how much life is left "
                "(dangerous — engine runs longer than expected). "
                "Points below 0 mean the model under-estimated (safe but wasteful)."
            )

            resid_model = st.selectbox(
                "Select model to inspect residuals",
                ["XGBoost", "LightGBM", "LSTM"],
                key="resid_model",
            )

            # Map the dropdown choice to the right column name
            resid_col_map = {
                "XGBoost":  "xgb_resid",
                "LightGBM": "lgbm_resid",
                "LSTM":     "lstm_resid",
            }
            resid_col = resid_col_map[resid_model]

            fig_resid = px.scatter(
                df.dropna(subset=[resid_col]),
                x="true_rul",
                y=resid_col,
                color_discrete_sequence=[MODEL_COLORS[resid_model]],
                title=f"{resid_model} — Residuals vs True RUL",
                labels={
                    "true_rul": "True RUL (cycles)",
                    resid_col:  "Residual: Predicted − True (cycles)",
                },
                hover_data=["engine_id", "true_rul", resid_col],
            )

            # Add a horizontal zero line — perfect predictions sit on this line
            fig_resid.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                annotation_text="Perfect prediction",
                annotation_position="top right",
            )

            st.plotly_chart(fig_resid, use_container_width=True)

            # -------------------------------------------------------------------
            # Section 2f — Head-to-head scatter: which model is closer per engine?
            # -------------------------------------------------------------------
            # For each engine, plot XGBoost error (x) vs LSTM error (y).
            # Points below the diagonal (y < x) = LSTM was more accurate.
            # Points above the diagonal (y > x) = XGBoost was more accurate.
            # This gives an intuitive view of where each model wins.
            # -------------------------------------------------------------------
            st.markdown("---")
            st.markdown("### Head-to-head: absolute error per engine")

            h2h_col1, h2h_col2 = st.columns(2)
            with h2h_col1:
                model_a = st.selectbox("Model A (X axis)", 
                                       ["XGBoost", "LightGBM", "LSTM"], 
                                       index=0, 
                                       key="h2h_a")
            with h2h_col2:
                model_b = st.selectbox("Model B (Y axis)", 
                                       ["XGBoost", "LightGBM", "LSTM"], 
                                       index=2, 
                                       key="h2h_b")

            # Map model names to their error columns
            err_col_map = {
                "XGBoost":  "xgb_err",
                "LightGBM": "lgbm_err",
                "LSTM":     "lstm_err",
            }

            a_col = err_col_map[model_a]
            b_col = err_col_map[model_b]

            h2h_df = df.dropna(subset=[a_col, b_col]).copy()

            # Label each point: which model won for that engine?
            h2h_df["winner"] = h2h_df.apply(
                lambda r: model_a 
                if r[a_col] < r[b_col] 
                else (model_b if r[b_col] < r[a_col] else "Tie"),
                axis=1,
            )

            fig_h2h = px.scatter(
                h2h_df,
                x=a_col,
                y=b_col,
                color="winner",
                color_discrete_map={
                    model_a: MODEL_COLORS[model_a],
                    model_b: MODEL_COLORS[model_b],
                    "Tie":   "gray",
                },
                hover_data=["engine_id", "true_rul"],
                title=f"Head-to-head: {model_a} error vs {model_b} error per engine",
                labels={
                    a_col: f"{model_a} Absolute Error (cycles)",
                    b_col: f"{model_b} Absolute Error (cycles)",
                },
            )

            # Add the diagonal line where both models perform equally well.
            # Points below = model_b wins, points above = model_a wins.
            max_err = max(h2h_df[a_col].max(), h2h_df[b_col].max()) * 1.05
            fig_h2h.add_shape(
                type="line", x0=0, y0=0, x1=max_err, y1=max_err,
                line=dict(color="gray", dash="dot"),
            )
            fig_h2h.add_annotation(
                x=max_err * 0.85, y=max_err * 0.75,
                text=f"← {model_b} wins",
                showarrow=False, font=dict(color="gray", size=11),
            )
            fig_h2h.add_annotation(
                x=max_err * 0.65, y=max_err * 0.92,
                text=f"{model_a} wins →",
                showarrow=False, font=dict(color="gray", size=11),
            )

            st.plotly_chart(fig_h2h, use_container_width=True)

            # Show win count summary below the chart
            win_summary = h2h_df["winner"].value_counts()
            w_cols = st.columns(3)
            w_cols[0].metric(f"{model_a} wins", int(win_summary.get(model_a, 0)))
            w_cols[1].metric("Tie", int(win_summary.get("Tie", 0)))
            w_cols[2].metric(f"{model_b} wins", int(win_summary.get(model_b, 0)))

            # -------------------------------------------------------------------
            # Section 2g — Raw results table (always at the bottom)
            # -------------------------------------------------------------------
            st.markdown("---")
            with st.expander("📋 View full results table"):
                display_cols = [
                    "engine_id", "engine_length", "true_rul", "rul_stage", "lifetime_group",
                    "xgb_pred", "lgbm_pred", "lstm_pred",
                    "xgb_err", "lgbm_err", "lstm_err",
                    "best_model",
                ]
                # Only show columns that actually exist (some may be None if API failed)
                display_cols = [c for c in display_cols if c in df.columns]
                st.dataframe(df[display_cols].round(2), use_container_width=True)

        else:
            # Friendly prompt if the user hasn't run predictions yet
            st.warning(
                "No results yet — click **Run All-Model Batch Predictions** above.",
                icon="⬆️",
            )

    except FileNotFoundError as e:
        st.error(
            f"Processed data not found: {e}\n\n"
            "Run the data pipeline first: `python -m src.data.load` → "
            "`preprocess` → `features.build_feature`"
        )