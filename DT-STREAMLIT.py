# app.py
# ==========================================================
# WIND TURBINE DIGITAL TWIN – THESIS DEMO (NO PLOTLY)
# - Simple UI (3 pages)
# - Local data shows Monthly + Daily graphs
# - Uploaded data hides Monthly/Daily graphs
# - ML page shows 3 Actual vs Predicted line plots (LR, DT, RF) with model name BELOW each graph
# - R² + RMSE combined bar chart (RMSE in yellow)
# ==========================================================

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# SETTINGS
# -----------------------------
st.set_page_config(page_title="Wind Turbine Digital Twin (Thesis Demo)", layout="wide")
REQUIRED_COLS = ["WindSpeed", "PowerOutput", "RotorSpeed", "PitchDeg", "offsetWindDirection"]

# -----------------------------
# HELPERS
# -----------------------------
def standardize_columns(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def validate_cols(df):
    return [c for c in REQUIRED_COLS if c not in df.columns]

def clean_df(df):
    df = df.copy()
    df = df.dropna(subset=REQUIRED_COLS)
    df = df[(df["WindSpeed"] >= 0) & (df["PowerOutput"] >= 0)]
    return df

@st.cache_data(show_spinner=False)
def train_models(df):
    X = df[["WindSpeed", "RotorSpeed", "PitchDeg", "offsetWindDirection"]]
    y = df["PowerOutput"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        results[name] = {
            "model": m,
            "r2": r2,
            "rmse": rmse,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    best_name = sorted(results.keys(), key=lambda k: results[k]["r2"], reverse=True)[0]
    return results, best_name

def plot_scatter(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["WindSpeed"], df["PowerOutput"], alpha=0.55)
    ax.set_title("Wind Speed vs Power Output (Hourly Data)")
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Power Output (kW)")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)

def plot_metrics(results):
    models = list(results.keys())
    r2_vals = [results[m]["r2"] for m in models]
    rmse_vals = [results[m]["rmse"] for m in models]

    x = np.arange(len(models))
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # R² (default color)
    ax1.bar(x - w/2, r2_vals, w, label="R²")

    # RMSE (yellow)
    ax2.bar(x + w/2, rmse_vals, w, label="RMSE (kW)", color="yellow")

    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("R² (-)")
    ax2.set_ylabel("RMSE (kW)")
    ax1.set_title("Model Performance (R² + RMSE)")
    ax1.grid(True, axis="y", alpha=0.25)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    st.pyplot(fig)

def plot_actual_vs_pred_line(y_test, y_pred, model_name, max_points=1400):
    """
    Line plot style like your screenshot.
    The x-axis is sample index (test set order). This can appear spiky, which is expected.
    """
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    n = min(len(y_test), len(y_pred), max_points)
    idx = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(idx, y_test[:n], linewidth=2, label="Actual Power")
    ax.plot(idx, y_pred[:n], linewidth=2, label=f"Predicted Power ({model_name})")

    ax.set_title("Actual vs Predicted Wind Power")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Power (kW)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    st.pyplot(fig)

    # Label below the graph (very clear for reviewers)
    st.markdown(
        f"<div style='text-align:center; font-weight:700; font-size:16px;'>Model: {model_name}</div>",
        unsafe_allow_html=True
    )

def run_optimization(best_model, df):
    ws = np.linspace(2, 15, 40)
    pitch = np.linspace(0, 30, 30)
    rotor = np.linspace(20, 200, 25)

    grid = pd.DataFrame([(a, b, c) for a in ws for b in pitch for c in rotor],
                        columns=["WindSpeed", "PitchDeg", "RotorSpeed"])

    grid["offsetWindDirection"] = float(df["offsetWindDirection"].mean())
    grid = grid[["WindSpeed", "RotorSpeed", "PitchDeg", "offsetWindDirection"]]
    grid["Pred_kW"] = best_model.predict(grid)

    best_point = grid.loc[grid["Pred_kW"].idxmax()]
    return best_point

# -----------------------------
# TITLE
# -----------------------------
st.title("🌬️ Wind Turbine Digital Twin - Hourly SCADA Data (2022)")
st.caption("Simple, defense-friendly: Hourly → Models → Optimization")

# -----------------------------
# SIDEBAR
# -----------------------------
page = st.sidebar.radio("Menu", ["Hourly Data", "ML Models", "Optimization"])

use_upload = st.sidebar.checkbox("Upload your own dataset (optional)", value=False)
uploaded = None
using_upload = False

if use_upload:
    uploaded = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        using_upload = True

# -----------------------------
# LOAD DATA
# -----------------------------
df = None
try:
    if uploaded is not None:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        df = standardize_columns(df)
        st.sidebar.success("Uploaded file loaded.")
    else:
        df = pd.read_excel("data_2022_hourly.xlsx")
        df = standardize_columns(df)
        st.sidebar.success("Loaded local hourly file.")
except Exception as e:
    st.error(f"Data loading error: {e}")
    st.stop()

missing = validate_cols(df)
if missing:
    st.error(f"Missing required columns: {missing}")
    st.info(f"Required columns: {REQUIRED_COLS}")
    st.stop()

df = clean_df(df)
if df.empty:
    st.error("After cleaning, dataset is empty.")
    st.stop()

st.success(f"✅ Loaded dataset successfully — {len(df):,} records after cleaning.")

# ==========================================================
# PAGE 1 — HOURLY DATA
# Local: show Monthly + Daily
# Upload: hide Monthly/Daily
# ==========================================================
if page == "Hourly Data":
    c1, c2, c3 = st.columns(3)
    c1.metric("Records (cleaned)", f"{len(df):,}")
    c2.metric("Avg Wind Speed (m/s)", f"{df['WindSpeed'].mean():.2f}")
    c3.metric("Avg Power (kW)", f"{df['PowerOutput'].mean():.2f}")

    st.subheader("Wind Speed vs Power Output (Hourly Data)")
    plot_scatter(df)

    st.subheader("Hourly Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True)

    # Only for LOCAL data: show Monthly & Daily graphs
    if not using_upload:
        st.markdown("---")
        st.subheader("Monthly and Daily Power Performance")

        try:
            monthly_df = pd.read_excel("data_2022_monthly.xlsx")
            daily_df = pd.read_excel("data_2022_daily.xlsx")
            monthly_df = standardize_columns(monthly_df)
            daily_df = standardize_columns(daily_df)

            monthly_df["Datetime"] = pd.to_datetime(monthly_df["Datetime"], errors="coerce")
            daily_df["Datetime"] = pd.to_datetime(daily_df["Datetime"], errors="coerce")

            # Monthly bar plot
            fig_m, ax_m = plt.subplots(figsize=(10, 5))
            ax_m.bar(monthly_df["Datetime"].dt.strftime("%b"), monthly_df["PowerOutput"])
            ax_m.set_xlabel("Month")
            ax_m.set_ylabel("Average Power Output (kW)")
            ax_m.set_title("Monthly Average Power Output (2022)")
            ax_m.grid(True, axis="y", alpha=0.25)
            st.pyplot(fig_m)

            # Daily line plot (selected month)
            selected_month = st.selectbox(
                "Select a month to view daily power output trend:",
                monthly_df["Datetime"].dt.strftime("%B")
            )
            month_num = monthly_df.loc[
                monthly_df["Datetime"].dt.strftime("%B") == selected_month, "Datetime"
            ].dt.month.values[0]

            daily_filtered = daily_df[daily_df["Datetime"].dt.month == month_num]

            fig_d, ax_d = plt.subplots(figsize=(10, 5))
            ax_d.plot(daily_filtered["Datetime"], daily_filtered["PowerOutput"], marker="o")
            ax_d.set_xlabel("Day")
            ax_d.set_ylabel("Average Power Output (kW)")
            ax_d.set_title(f"Daily Average Power Output – {selected_month} 2022")
            ax_d.grid(True, alpha=0.25)
            st.pyplot(fig_d)

        except Exception as e:
            st.warning(f"Monthly/Daily files not available: {e}")

    else:
        st.info("📌 Upload mode: Monthly/Daily graphs are hidden (as requested).")

# ==========================================================
# PAGE 2 — ML MODELS
# - Performance bar chart
# - 3 Actual vs Predicted line plots (LR, DT, RF)
# ==========================================================
elif page == "ML Models":
    st.subheader("Train & Compare ML Models")

    results, best_name = train_models(df)
    st.success(f"Best model: **{best_name}** (highest R²)")

    score_table = pd.DataFrame({
        "Model": list(results.keys()),
        "R²": [results[m]["r2"] for m in results],
        "RMSE (kW)": [results[m]["rmse"] for m in results],
    }).sort_values("R²", ascending=False)
    st.dataframe(score_table, use_container_width=True)

    st.subheader("Model Performance (R² + RMSE)")
    plot_metrics(results)

    st.markdown("---")
    st.subheader("Actual vs Predicted Wind Power (All 3 Models)")

    max_points = st.slider(
        "Number of points to display (for clear plot)",
        min_value=300,
        max_value=2000,
        value=1400,
        step=100
    )

    # Show 3 graphs stacked (simple + clear)
    for model_name in ["Linear Regression", "Decision Tree", "Random Forest"]:
        r = results[model_name]
        st.markdown("### 📉 Actual vs Predicted Wind Power (Line Plot)")
        plot_actual_vs_pred_line(
            y_test=r["y_test"],
            y_pred=r["y_pred"],
            model_name=model_name,
            max_points=max_points
        )
        st.markdown("---")

# ==========================================================
# PAGE 3 — OPTIMIZATION
# ==========================================================
elif page == "Optimization":
    st.subheader("Optimization (Digital Twin Simulation)")

    results, best_name = train_models(df)
    best_model = results[best_name]["model"]

    if st.button("Run Optimization (Grid Search)"):
        with st.spinner("Running optimization..."):
            best_point = run_optimization(best_model, df)
        st.session_state["best_point"] = best_point
        st.success("Optimization complete.")

    best_point = st.session_state.get("best_point", None)
    if best_point is not None:
        st.markdown("### Optimal Operating Point")
        st.write(f"- Wind Speed: **{best_point['WindSpeed']:.2f} m/s**")
        st.write(f"- Pitch: **{best_point['PitchDeg']:.2f}°**")
        st.write(f"- Rotor Speed: **{best_point['RotorSpeed']:.2f} RPM**")
        st.write(f"- Predicted Power: **{best_point['Pred_kW']:.2f} kW**")
    else:
        st.info("Click **Run Optimization** to compute the best operating point.")

    st.markdown("---")
    st.subheader("Interactive Power Prediction (Best Model)")

    colA, colB, colC = st.columns(3)
    ws = colA.slider("Wind Speed (m/s)", 0.0, 25.0, 7.0, 0.1)
    pitch = colB.slider("Pitch (°)", 0.0, 30.0, 5.0, 0.5)
    rotor = colC.slider("Rotor Speed (RPM)", 0.0, 300.0, 100.0, 1.0)

    X_in = pd.DataFrame([{
        "WindSpeed": ws,
        "RotorSpeed": rotor,
        "PitchDeg": pitch,
        "offsetWindDirection": float(df["offsetWindDirection"].mean())
    }])

    pred_kw = float(best_model.predict(X_in)[0])
    st.success(f"Predicted Power Output: **{pred_kw:.2f} kW** (Model: {best_name})")

st.caption("Developed by Ajaykumar Pashikanti • Digital Twin Framework (Hourly + ML + Optimization)")
