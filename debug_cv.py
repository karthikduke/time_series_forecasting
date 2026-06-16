"""Quick diagnostic to see what Prophet cross_validation actually returns."""
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# Load & prep
df = pd.read_csv(r"MT_Incoming_2026.csv")
df["ds"] = pd.to_datetime(df["ds"], format="%d-%b-%y")
df = df.dropna(subset=["y"])
df["y"] = df["y"].astype(float)

# Log transform
df_log = df[["ds", "y"]].copy()
df_log["y"] = np.log1p(df_log["y"])
df_log["floor"] = 0.0
df_log["cap"] = float(np.log1p(df["y"].max() * 3))
df_log["is_weekday"] = (df_log["ds"].dt.dayofweek < 5).astype(float)

print(f"Data: {len(df_log)} rows, cap={df_log['cap'].iloc[0]:.4f}")
print(f"y range (log): {df_log['y'].min():.4f} to {df_log['y'].max():.4f}")
print()

# Fit a simple model
m = Prophet(
    growth="logistic",
    changepoint_prior_scale=0.3,
    seasonality_mode="multiplicative",
    weekly_seasonality=True,
    yearly_seasonality=False,
    daily_seasonality=False,
    interval_width=0.80,
)
m.add_regressor("is_weekday", mode="multiplicative")
m.fit(df_log[["ds", "y", "is_weekday", "floor", "cap"]])

# Run CV
print("Running cross_validation...")
try:
    df_cv = cross_validation(m, initial="73 days", period="14 days", horizon="7 days")
    print(f"CV result shape: {df_cv.shape}")
    print(f"CV columns: {list(df_cv.columns)}")
    print(f"\ndf_cv head:")
    print(df_cv.head(10).to_string())
    print(f"\ndf_cv describe:")
    print(df_cv[["y", "yhat"]].describe().to_string())
    print(f"\nAny NaN in yhat? {df_cv['yhat'].isna().sum()}")
    print(f"Any inf in yhat? {np.isinf(df_cv['yhat']).sum()}")
    print(f"Any NaN in y? {df_cv['y'].isna().sum()}")

    # Manual metrics
    errors = df_cv["y"] - df_cv["yhat"]
    print(f"\nManual MAE: {errors.abs().mean():.6f}")
    print(f"Manual RMSE: {np.sqrt((errors**2).mean()):.6f}")
    print(f"Any inf errors: {np.isinf(errors).sum()}")
    print(f"Max abs error: {errors.abs().max():.6f}")

    # Prophet's metrics
    metrics = performance_metrics(df_cv, rolling_window=1)
    print(f"\nProphet metrics columns: {list(metrics.columns)}")
    print(f"Prophet RMSE: {metrics['rmse'].values}")
    print(f"Prophet MAE: {metrics['mae'].values}")
    print(f"Any NaN in RMSE column? {metrics['rmse'].isna().sum()}")
    print(f"Any inf in RMSE column? {np.isinf(metrics['rmse']).sum()}")

except Exception as e:
    print(f"CV FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
