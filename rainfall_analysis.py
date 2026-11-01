import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(file_name="rainfall_area-wt_sd_1901-2015.csv"):
    df = pd.read_csv(file_name)
    df.columns = df.columns.str.strip()

    year_col = next((c for c in df.columns if c.upper() == "YEAR"), None)
    annual_col = next((c for c in df.columns if c.upper() == "ANNUAL"), None)
    region_col = next(
        (c for c in df.columns
         if c.upper().replace("_", "").replace(" ", "") == "SUBDIVISION"),
        None
    )

    if year_col is None or annual_col is None or region_col is None:
        raise ValueError("CSV needs YEAR, SUBDIVISION and ANNUAL columns.")

    df = df.rename(columns={
        year_col: "YEAR",
        annual_col: "ANNUAL",
        region_col: "SUBDIVISION"
    })
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df["ANNUAL"] = pd.to_numeric(df["ANNUAL"], errors="coerce")
    df["SUBDIVISION"] = df["SUBDIVISION"].astype(str).str.strip()
    df = df.dropna(subset=["YEAR", "ANNUAL", "SUBDIVISION"])
    df = df[df["SUBDIVISION"] != ""]
    df["YEAR"] = df["YEAR"].astype(int)
    return df.sort_values(["SUBDIVISION", "YEAR"]).reset_index(drop=True)


def prepare_region(df, region):
    data = df[df["SUBDIVISION"] == region].copy()
    data = data.sort_values("YEAR").reset_index(drop=True)
    data["PREVIOUS_YEAR"] = data["ANNUAL"].shift(1)
    data["TWO_YEARS_AGO"] = data["ANNUAL"].shift(2)
    data["THREE_YEARS_AGO"] = data["ANNUAL"].shift(3)
    return data


def get_outliers(data):
    q1 = data["ANNUAL"].quantile(0.25)
    q3 = data["ANNUAL"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = data[(data["ANNUAL"] < lower) | (data["ANNUAL"] > upper)]
    return outliers, lower, upper


def train_models(data):
    data = data.sort_values("YEAR").copy()
    if len(data) < 10:
        raise ValueError("At least 10 yearly records are needed.")

    split = int(len(data) * 0.8)
    train = data.iloc[:split]
    test = data.iloc[split:]
    X_train = train[["YEAR"]]
    X_test = test[["YEAR"]]
    y_train = train["ANNUAL"]
    y_test = test["ANNUAL"]
    results = {}

    # 1. Simple Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results["Simple Linear Regression"] = {
        "model": model, "mae": mean_absolute_error(y_test, pred),
        "mse": mean_squared_error(y_test, pred),
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "r2": r2_score(y_test, pred)
    }

    # 2. Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    pred = model.predict(X_test_poly)
    results["Polynomial Regression"] = {
        "model": model, "polynomial": poly,
        "mae": mean_absolute_error(y_test, pred),
        "mse": mean_squared_error(y_test, pred),
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "r2": r2_score(y_test, pred)
    }
 # Prepare previous-year features for models 3, 4 and 5
    lag_data = data.dropna(
        subset=["PREVIOUS_YEAR", "TWO_YEARS_AGO", "THREE_YEARS_AGO"]
    ).copy()
    lag_split = int(len(lag_data) * 0.8)
    lag_train = lag_data.iloc[:lag_split]
    lag_test = lag_data.iloc[lag_split:]
    features = ["PREVIOUS_YEAR", "TWO_YEARS_AGO", "THREE_YEARS_AGO"]
    X_lag_train = lag_train[features]
    X_lag_test = lag_test[features]
    y_lag_train = lag_train["ANNUAL"]
    y_lag_test = lag_test["ANNUAL"]

    # 3. Multiple Linear Regression
    model = LinearRegression()
    model.fit(X_lag_train, y_lag_train)
    pred = model.predict(X_lag_test)
    results["Multiple Linear Regression"] = {
        "model": model, "mae": mean_absolute_error(y_lag_test, pred),
        "mse": mean_squared_error(y_lag_test, pred),
        "rmse": np.sqrt(mean_squared_error(y_lag_test, pred)),
        "r2": r2_score(y_lag_test, pred)
    }

    # 4. Decision Tree Regression
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X_lag_train, y_lag_train)
    pred = model.predict(X_lag_test)
    results["Decision Tree Regression"] = {
        "model": model, "mae": mean_absolute_error(y_lag_test, pred),
        "mse": mean_squared_error(y_lag_test, pred),
        "rmse": np.sqrt(mean_squared_error(y_lag_test, pred)),
        "r2": r2_score(y_lag_test, pred)
    }

    # 5. Random Forest Regression
    model = RandomForestRegressor(
        n_estimators=100, max_depth=5, random_state=42
    )
    model.fit(X_lag_train, y_lag_train)
    pred = model.predict(X_lag_test)
    results["Random Forest Regression"] = {
        "model": model, "mae": mean_absolute_error(y_lag_test, pred),
        "mse": mean_squared_error(y_lag_test, pred),
        "rmse": np.sqrt(mean_squared_error(y_lag_test, pred)),
        "r2": r2_score(y_lag_test, pred)
    }
    return results


def predict_next_year(data, results):
    data = data.sort_values("YEAR")
    next_year = int(data["YEAR"].max()) + 1

    simple = results["Simple Linear Regression"]["model"].predict(
        pd.DataFrame({"YEAR": [next_year]})
    )[0]

    poly_info = results["Polynomial Regression"]
    poly_x = poly_info["polynomial"].transform(
        pd.DataFrame({"YEAR": [next_year]})
    )
    polynomial = poly_info["model"].predict(poly_x)[0]

    last_three = data["ANNUAL"].tail(3).tolist()
    lag_input = pd.DataFrame({
        "PREVIOUS_YEAR": [last_three[-1]],
        "TWO_YEARS_AGO": [last_three[-2]],
        "THREE_YEARS_AGO": [last_three[-3]]
    })

    predictions = {
        "Year": next_year,
        "Simple Linear Regression": max(0, simple),
        "Polynomial Regression": max(0, polynomial)
    }

    for name in ["Multiple Linear Regression",
                 "Decision Tree Regression", "Random Forest Regression"]:
        predictions[name] = max(0, results[name]["model"].predict(lag_input)[0])

    return predictions
