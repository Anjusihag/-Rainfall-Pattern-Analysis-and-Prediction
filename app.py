import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="Rainfall Pattern Analysis",
    page_icon="🌧️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
  background: radial-gradient(circle at 12% 8%, rgba(0,210,255,.20), transparent 28%),
              radial-gradient(circle at 88% 18%, rgba(167,89,255,.24), transparent 30%),
              linear-gradient(135deg,#06152f 0%,#0d2c55 43%,#291c50 76%,#101a3b 100%);
  background-attachment: fixed; color: #f4f7ff;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#071a33 0%,#0b3150 55%,#102d48 100%);
  border-right: 1px solid rgba(91, 220, 230, .22);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
[data-testid="stSidebar"] [data-testid="stButton"] { margin: 8px 0; }
[data-testid="stSidebar"] [data-testid="stButton"] button {
  width: 100%;
  min-height: 52px;
  justify-content: flex-start;
  text-align: left;
  padding: 0.75rem 1rem;
  border-radius: 14px;
  border: 1px solid rgba(117, 204, 220, .20);
  background: linear-gradient(110deg, rgba(18, 58, 83, .96), rgba(19, 48, 76, .96));
  color: #d9f3f6 !important;
  font-size: 15px;
  font-weight: 650;
  letter-spacing: .2px;
  box-shadow: 0 5px 14px rgba(0,0,0,.13);
  transition: all .18s ease-in-out;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
  border-color: #54dce0;
  background: linear-gradient(110deg, #126b79, #164f70);
  color: #ffffff !important;
  transform: translateX(3px);
  box-shadow: 0 7px 18px rgba(0, 190, 205, .18);
}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
  background: linear-gradient(110deg, #13a9b5, #1673a5);
  color: #ffffff !important;
  border: 1px solid rgba(135, 247, 245, .8);
  box-shadow: 0 7px 20px rgba(0, 190, 205, .24);
}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]:hover {
  background: linear-gradient(110deg, #18bac2, #2189ba);
}
h1,h2,h3,h4,p,label { color:#f4f7ff !important; }
h1 { background:linear-gradient(90deg,#70e4ff,#c2a2ff,#ff9acb); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
div[data-testid="stMetric"] {
  background:linear-gradient(120deg,rgba(0,116,255,.82),rgba(99,47,235,.86));
  border:1px solid rgba(174,211,255,.35); padding:17px; border-radius:17px;
  box-shadow:0 8px 25px rgba(0,0,0,.18);
}
div[data-testid="stMetricLabel"] { color:#d7e7ff !important; }
div[data-testid="stMetricValue"] { color:#fff !important; }
.stButton>button,.stDownloadButton>button { background:linear-gradient(90deg,#09bfe9,#743cff); color:white; border:0; border-radius:11px; }
</style>
""", unsafe_allow_html=True)

st.title("🌧️ Rainfall Pattern Analysis & Prediction")
st.caption("Explore historical rainfall data, compare machine-learning models, and inspect unusual rainfall values.")

# --------------------------------------------------
# Load and clean data
# --------------------------------------------------
@st.cache_data
def load_rainfall_data(uploaded_file=None):
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
    else:
        # Keep the CSV file in the same folder as app.py.
        raw = pd.read_csv("rainfall_area-wt_sd_1901-2015.csv")

    raw.columns = [str(c).strip().upper() for c in raw.columns]

    # Identify important columns even if their names vary slightly.
    year_col = next((c for c in raw.columns if c == "YEAR" or "YEAR" in c), None)
    subdivision_col = next((c for c in raw.columns if "SUBDIVISION" in c or "REGION" in c or "STATE" in c), None)
    annual_col = next((c for c in raw.columns if c == "ANNUAL" or "ANNUAL" in c), None)

    if year_col is None or annual_col is None:
        raise ValueError("Could not find YEAR and ANNUAL columns in the CSV file.")

    rename_map = {year_col: "YEAR", annual_col: "ANNUAL"}
    if subdivision_col is not None:
        rename_map[subdivision_col] = "SUBDIVISION"
    raw = raw.rename(columns=rename_map)

    if "SUBDIVISION" not in raw.columns:
        raw["SUBDIVISION"] = "All India / Overall"

    raw["YEAR"] = pd.to_numeric(raw["YEAR"], errors="coerce")
    raw["ANNUAL"] = pd.to_numeric(raw["ANNUAL"], errors="coerce")
    raw = raw.dropna(subset=["YEAR", "ANNUAL"])
    raw["YEAR"] = raw["YEAR"].astype(int)
    raw["SUBDIVISION"] = raw["SUBDIVISION"].astype(str).str.strip()

    # Convert rainfall month columns to numbers when present.
    month_names = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                   "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    for month in month_names:
        possible = [c for c in raw.columns if c == month or c.startswith(month + "_")]
        for col in possible:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    return raw.sort_values(["SUBDIVISION", "YEAR"]).reset_index(drop=True)

st.sidebar.markdown(
    "<div style='font-size:22px;font-weight:800;color:#35caff;padding:8px 0 16px;'>🌧️ Rainfall Analysis</div>",
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

if "active_page" not in st.session_state:
    st.session_state.active_page = "Dashboard"

nav_items = [
    ("Dashboard", "🏠  Dashboard"),
    ("Rainfall Explorer", "📊  Rainfall Explorer"),
    ("Prediction Lab", "🧠  Prediction Lab"),
    ("Anomaly Analysis", "🚨  Anomaly Analysis"),
]
for page_name, button_label in nav_items:
    is_active = st.session_state.active_page == page_name
    if st.sidebar.button(
        button_label,
        key="nav_" + page_name,
        use_container_width=True,
        type="primary" if is_active else "secondary"
    ):
        st.session_state.active_page = page_name

st.sidebar.markdown("---")
try:
    df = load_rainfall_data()
except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.info("Keep rainfall_area-wt_sd_1901-2015.csv in the same folder as app.py.")
    st.stop()

page = st.session_state.active_page
regions = sorted(df["SUBDIVISION"].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("Choose subdivision / region", regions)
region_data = df[df["SUBDIVISION"] == selected_region].sort_values("YEAR").copy()

if region_data.empty:
    st.warning("No records are available for this selection.")
    st.stop()

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def add_matplotlib_chart(fig):
    """Apply consistent dark styling to every Matplotlib chart in every section."""
    fig.patch.set_facecolor("#0d2345")
    for ax in fig.axes:
        ax.set_facecolor("#11294c")
        ax.tick_params(colors="#eaf2ff", labelcolor="#eaf2ff")
        ax.xaxis.label.set_color("#d9e7ff")
        ax.yaxis.label.set_color("#d9e7ff")
        ax.title.set_color("#ffffff")
        for spine in ax.spines.values():
            spine.set_color("#31517b")
        ax.grid(True, color="#8da7d1", alpha=0.16)
        ax.set_axisbelow(True)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("#172f53")
            legend.get_frame().set_edgecolor("#5476a5")
            for item in legend.get_texts():
                item.set_color("#f0f5ff")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def get_iqr_limits(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if values.empty:
        return np.nan, np.nan, pd.Series(dtype=float)
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr, values

def make_model_results(data):
    """Train five regression models using earlier years for training and later years for testing."""
    clean = data[["YEAR", "ANNUAL"]].dropna().sort_values("YEAR").copy()
    if len(clean) < 10:
        return {}, None, None, "At least 10 annual records are recommended to compare models."

    X_year = clean[["YEAR"]].to_numpy()
    y = clean["ANNUAL"].to_numpy()
    split = max(2, int(len(clean) * 0.8))
    if split >= len(clean):
        split = len(clean) - 1

    X_train, X_test = X_year[:split], X_year[split:]
    y_train, y_test = y[:split], y[split:]

    models = {
        "Simple Linear Regression": LinearRegression(),
        "Polynomial Regression": make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression()),
        "Decision Tree Regression": DecisionTreeRegressor(max_depth=3, random_state=42),
        "Random Forest Regression": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "model": model,
            "MAE": mean_absolute_error(y_test, pred),
            "MSE": mean_squared_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred) if len(y_test) > 1 else np.nan
        }

    # Multiple linear regression uses the previous three years as features.
    lag_data = clean.copy()
    lag_data["LAG_1"] = lag_data["ANNUAL"].shift(1)
    lag_data["LAG_2"] = lag_data["ANNUAL"].shift(2)
    lag_data["LAG_3"] = lag_data["ANNUAL"].shift(3)
    lag_data = lag_data.dropna()
    if len(lag_data) >= 8:
        X_lag = lag_data[["LAG_1", "LAG_2", "LAG_3"]].to_numpy()
        y_lag = lag_data["ANNUAL"].to_numpy()
        lag_split = max(2, int(len(lag_data) * 0.8))
        if lag_split >= len(lag_data):
            lag_split = len(lag_data) - 1
        lag_model = LinearRegression()
        lag_model.fit(X_lag[:lag_split], y_lag[:lag_split])
        lag_pred = lag_model.predict(X_lag[lag_split:])
        results["Multiple Linear Regression"] = {
            "model": lag_model,
            "MAE": mean_absolute_error(y_lag[lag_split:], lag_pred),
            "MSE": mean_squared_error(y_lag[lag_split:], lag_pred),
            "RMSE": np.sqrt(mean_squared_error(y_lag[lag_split:], lag_pred)),
            "R²": r2_score(y_lag[lag_split:], lag_pred) if len(lag_pred) > 1 else np.nan
        }

    return results, X_train, y_train, None

# --------------------------------------------------
# 1. Dashboard
# --------------------------------------------------
if page == "Dashboard":
    st.markdown(
        "<div style='padding:4px 0 12px 0;'>"
        "<div style='font-size:30px;font-weight:800;color:#f7f9ff;'>🌧️ Rainfall Pattern Analysis and Prediction</div>"
        "<div style='font-size:15px;color:#c9dcff;'>Explore Historical Indian Rainfall | Compare Machine Learning Models | Predict Future Rainfall</div>"
        "</div>", unsafe_allow_html=True
    )

    first_year, last_year = int(df["YEAR"].min()), int(df["YEAR"].max())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Subdivisions", f"{df['SUBDIVISION'].nunique()}")
    c3.metric("First Year", str(first_year))
    c4.metric("Last Year", str(last_year))

    chart_bg, panel_bg, text_color = "#0d2345", "#11294c", "#eaf2ff"

    def polish(ax, title, xlabel=None, ylabel=None):
        ax.set_facecolor(panel_bg)
        ax.set_title(title, color=text_color, fontsize=11, fontweight="bold", pad=9)
        if xlabel: ax.set_xlabel(xlabel, color=text_color)
        if ylabel: ax.set_ylabel(ylabel, color=text_color)
        ax.tick_params(colors=text_color, labelsize=8)
        for spine in ax.spines.values(): spine.set_color("#31517b")
        ax.grid(True, color="#7d9ac7", alpha=.15)
        ax.set_axisbelow(True)

    annual_mean = df.groupby("YEAR", as_index=False)["ANNUAL"].mean()
    left, right = st.columns([1.15, .95], gap="medium")
    with left:
        st.markdown("### 📈 Annual Rainfall Trend")
        fig, ax = plt.subplots(figsize=(8, 3.4))
        fig.patch.set_facecolor(chart_bg)
        ax.plot(annual_mean["YEAR"], annual_mean["ANNUAL"], color="#35e0dc", linewidth=2.4,
                marker="o", markersize=3, markerfacecolor="#a9fff2", markeredgecolor="#fff")
        ax.fill_between(annual_mean["YEAR"], annual_mean["ANNUAL"], color="#16c9d4", alpha=.20)
        polish(ax, "Mean Annual Rainfall Across Available Subdivisions", "Year", "Rainfall (mm)")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown("### 📊 Average Rainfall by Subdivision")
        ranking = df.groupby("SUBDIVISION")["ANNUAL"].mean().sort_values(ascending=False).head(10).sort_values()
        fig, ax = plt.subplots(figsize=(7, 3.4))
        fig.patch.set_facecolor(chart_bg)
        colors = plt.cm.plasma(np.linspace(.35, .95, len(ranking)))
        bars = ax.barh(ranking.index, ranking.values, color=colors, edgecolor="#ffe3ff", linewidth=.35)
        polish(ax, "Top 10 Highest Average Annual Rainfall", "Average Rainfall (mm)")
        ax.tick_params(axis="y", labelsize=7)
        ax.bar_label(bars, fmt="%.0f", padding=3, color=text_color, fontsize=7)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    bottom1, bottom2 = st.columns(2, gap="medium")
    month_names = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    month_cols = []
    for month in month_names:
        col = next((c for c in df.columns if c == month or c.startswith(month + "_")), None)
        if col is not None: month_cols.append((month.title(), col))

    with bottom1:
        st.markdown("### 🌈 Monthly Rainfall Pattern")
        if month_cols:
            month_means = [df[col].mean() for _, col in month_cols]
            fig, ax = plt.subplots(figsize=(7, 3.2))
            fig.patch.set_facecolor(chart_bg)
            colors = plt.cm.turbo(np.linspace(.05, .90, len(month_cols)))
            bars = ax.bar([m for m, _ in month_cols], month_means, color=colors, edgecolor="#e9f5ff", linewidth=.35)
            polish(ax, "Average Monthly Rainfall Across Subdivisions", "Month", "Rainfall (mm)")
            ax.bar_label(bars, fmt="%.0f", padding=2, color=text_color, fontsize=7)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Monthly columns were not found in this dataset.")

    with bottom2:
        st.markdown("### 🥧 Seasonal Rainfall Contribution")
        if month_cols:
            month_average = {m.upper(): float(df[col].mean()) for m, col in month_cols}
            seasons = {
                "SW Monsoon (Jun–Sep)": ["JUN","JUL","AUG","SEP"],
                "Pre-Monsoon (Mar–May)": ["MAR","APR","MAY"],
                "Post-Monsoon (Oct–Dec)": ["OCT","NOV","DEC"],
                "Winter (Jan–Feb)": ["JAN","FEB"]
            }
            season_values = {s: sum(month_average.get(m, 0) for m in months) for s, months in seasons.items()}
            season_values = {k:v for k,v in season_values.items() if v > 0}
            if season_values:
                fig, ax = plt.subplots(figsize=(7, 3.2))
                fig.patch.set_facecolor(chart_bg)
                wedges, _, _ = ax.pie(
                    list(season_values.values()), labels=None, autopct="%1.1f%%", startangle=90,
                    colors=["#20c997","#ffb347","#c65cff","#21c8f6"][:len(season_values)],
                    wedgeprops={"width":.48,"edgecolor":chart_bg,"linewidth":2},
                    textprops={"color":"white","fontsize":8}
                )
                ax.legend(wedges, list(season_values.keys()), loc="center left", bbox_to_anchor=(.86,.5),
                          frameon=False, labelcolor=text_color, fontsize=7)
                ax.set_title("Share of Average Monthly Rainfall by Season", color=text_color, fontsize=11, fontweight="bold")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        else:
            st.info("Seasonal chart requires monthly rainfall columns.")

    st.caption("Historical rainfall summary. Seasonal contribution is descriptive and is not a flood-risk prediction.")

# --------------------------------------------------
# 2. Rainfall Explorer
# --------------------------------------------------
elif page == "Rainfall Explorer":
    st.header("Rainfall Explorer")
    min_year, max_year = int(region_data["YEAR"].min()), int(region_data["YEAR"].max())
    if min_year < max_year:
        year_range = st.slider("Choose year range", min_year, max_year, (min_year, max_year))
    else:
        year_range = (min_year, max_year)

    filtered = region_data[region_data["YEAR"].between(year_range[0], year_range[1])].copy()

    st.subheader("Annual Rainfall for Selected Years")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(filtered["YEAR"], filtered["ANNUAL"], color="#39e0d0", marker="o",
            markerfacecolor="#ff9bd5", markeredgecolor="#ffffff", markersize=4, linewidth=2.2)
    ax.fill_between(filtered["YEAR"], filtered["ANNUAL"], color="#39e0d0", alpha=0.18)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Rainfall (mm)")
    ax.set_title(f"Rainfall from {year_range[0]} to {year_range[1]}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    add_matplotlib_chart(fig)

    month_names = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                   "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    month_cols = []
    for month in month_names:
        col = next((c for c in filtered.columns if c == month or c.startswith(month + "_")), None)
        if col is not None:
            month_cols.append((month, col))

    if month_cols:
        month_avg = [filtered[col].mean() for _, col in month_cols]
        st.subheader("Average Monthly Rainfall")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        month_colors = plt.cm.turbo(np.linspace(0.05, 0.92, len(month_cols)))
        ax.bar([m for m, _ in month_cols], month_avg, color=month_colors,
               edgecolor="#eaf2ff", linewidth=0.45)
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Rainfall (mm)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        add_matplotlib_chart(fig)
    else:
        st.info("Monthly chart is unavailable because monthly rainfall columns were not found in this dataset.")

    st.subheader("Filtered Data")
    st.dataframe(filtered, use_container_width=True)
    st.download_button(
        "Download filtered data as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_rainfall_data.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# 3. Prediction Lab
# --------------------------------------------------
elif page == "Prediction Lab":
    st.header("Prediction Lab")
    st.write("Compare five regression models using historical annual rainfall. The predictions are educational estimates, not official weather forecasts.")

    results, _, _, error = make_model_results(region_data)
    if error:
        st.warning(error)
    else:
        metric_rows = []
        for name, result in results.items():
            metric_rows.append({
                "Model": name,
                "MAE": round(result["MAE"], 2),
                "MSE": round(result["MSE"], 2),
                "RMSE": round(result["RMSE"], 2),
                "R²": round(result["R²"], 3) if pd.notna(result["R²"]) else "N/A"
            })
        metrics_df = pd.DataFrame(metric_rows).sort_values("RMSE")
        st.subheader("Model Performance")
        st.dataframe(metrics_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        model_colors = plt.cm.plasma(np.linspace(0.25, 0.9, len(metrics_df)))
        bars = ax.bar(metrics_df["Model"], metrics_df["RMSE"], color=model_colors,
                      edgecolor="#ffffff", linewidth=0.55)
        ax.bar_label(bars, fmt="%.1f", padding=3, color="#f4f7ff", fontsize=8)
        ax.set_title("Model Comparison by RMSE (lower is better)")
        ax.set_ylabel("RMSE (mm)")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        add_matplotlib_chart(fig)

        next_year = int(region_data["YEAR"].max()) + 1
        st.subheader(f"Estimated Rainfall for {next_year}")
        predictions = []
        for name, result in results.items():
            model = result["model"]
            try:
                if name == "Multiple Linear Regression":
                    last_values = region_data.sort_values("YEAR")["ANNUAL"].tail(3).to_numpy()
                    if len(last_values) == 3:
                        pred_value = float(model.predict([[last_values[-1], last_values[-2], last_values[-3]]])[0])
                    else:
                        continue
                else:
                    pred_value = float(model.predict(np.array([[next_year]]))[0])
                predictions.append({"Model": name, "Estimated rainfall (mm)": round(max(0, pred_value), 2)})
            except Exception:
                pass

        if predictions:
            prediction_df = pd.DataFrame(predictions)
            st.dataframe(prediction_df, use_container_width=True)
            fig, ax = plt.subplots(figsize=(9, 4))
            pred_colors = plt.cm.turbo(np.linspace(0.12, 0.88, len(prediction_df)))
            bars = ax.bar(prediction_df["Model"], prediction_df["Estimated rainfall (mm)"],
                          color=pred_colors, edgecolor="#ffffff", linewidth=0.55)
            ax.bar_label(bars, fmt="%.0f", padding=3, color="#f4f7ff", fontsize=8)
            ax.set_ylabel("Estimated rainfall (mm)")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            add_matplotlib_chart(fig)

# --------------------------------------------------
# 4. Anomaly Analysis
# --------------------------------------------------
elif page == "Anomaly Analysis":
    st.header("Anomaly Analysis")
    st.write("The IQR method marks unusually low or high annual rainfall values. An anomaly is not automatically an error or a flood.")

    lower, upper, values = get_iqr_limits(region_data["ANNUAL"])
    if values.empty:
        st.warning("No annual rainfall values are available for anomaly analysis.")
    else:
        anomalies = region_data[(region_data["ANNUAL"] < lower) | (region_data["ANNUAL"] > upper)].copy()

        c1, c2, c3 = st.columns(3)
        c1.metric("Lower IQR Limit", f"{lower:,.1f} mm")
        c2.metric("Upper IQR Limit", f"{upper:,.1f} mm")
        c3.metric("Anomalies Found", str(len(anomalies)))

        st.subheader("Rainfall Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(
            region_data["ANNUAL"].dropna(), vert=False, patch_artist=True,
            boxprops=dict(facecolor="#7357d9", color="#b7a0ff"),
            medianprops=dict(color="#ff83c8", linewidth=2.2),
            whiskerprops=dict(color="#39d9ff", linewidth=1.5),
            capprops=dict(color="#39d9ff", linewidth=1.5),
            flierprops=dict(marker="o", markerfacecolor="#ffb86b",
                            markeredgecolor="#ffffff", alpha=0.85)
        )
        ax.set_xlabel("Annual Rainfall (mm)")
        ax.set_title("Boxplot of Annual Rainfall")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        add_matplotlib_chart(fig)

        st.subheader("Annual Rainfall and Anomalies")
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.scatter(region_data["YEAR"], region_data["ANNUAL"], s=30, color="#39d9ff",
                   edgecolor="#ffffff", linewidth=0.25, label="Annual rainfall")
        if not anomalies.empty:
            ax.scatter(anomalies["YEAR"], anomalies["ANNUAL"], s=90, color="#ff5ca8",
                       marker="X", edgecolor="#ffffff", linewidth=0.7, label="IQR anomaly")
        ax.axhline(lower, color="#ffb86b", linestyle="--", linewidth=1.6, label="Lower IQR limit")
        ax.axhline(upper, color="#c79bff", linestyle="--", linewidth=1.6, label="Upper IQR limit")
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual Rainfall (mm)")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        add_matplotlib_chart(fig)

        st.subheader("Anomaly Records")
        if anomalies.empty:
            st.success("No IQR anomalies were detected for this region.")
        else:
            st.dataframe(anomalies[["YEAR", "SUBDIVISION", "ANNUAL"]], use_container_width=True)
            st.download_button(
                "Download anomaly records",
                data=anomalies.to_csv(index=False).encode("utf-8"),
                file_name="rainfall_anomalies.csv",
                mime="text/csv"
            )

st.markdown("---")
st.caption("Rainfall analysis project • Matplotlib charts • Streamlit interface")
