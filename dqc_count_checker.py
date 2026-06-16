"""
DQC Count Deviation Checker - Improved Prophet-based anomaly detection.

Fixes for the negative lower-bound problem:
1. Log-transform (log1p/expm1) ensures predictions stay >= 0
2. Logistic growth with floor=0 enforces non-negativity
3. Aggressive changepoint detection catches trend shifts early
4. Weekend/holiday regressor separates zero-days from business days
5. Multi-signal anomaly detection (not just a single threshold)

v2 Enhancements:
  - Configurable growth type, log transform, cap multiplier
  - Holiday calendar support (country + custom business calendars)
  - Tunable weekly/yearly/monthly seasonality orders
  - `from_tuning_json()` to load parameters from the hyperparameter tuner

Usage:
    checker = DQCCountDeviationChecker()
    checker.fit_and_predict(df_historical)   # df with columns: ds, y
    result = checker.check(today_date, today_count, df_recent_history)

    if result['status'] in ('WARNING', 'CRITICAL'):
        send_alert(result['message'])

    # With tuned parameters from the hyperparameter tuner:
    config = CheckerConfig.from_tuning_json("tuning_results_XXX.json")
    checker = DQCCountDeviationChecker(config)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from prophet import Prophet

logger = logging.getLogger(__name__)


@dataclass
class CheckerConfig:
    """
    Configuration for the DQC Count Deviation Checker.

    All Prophet model parameters can be loaded from the hyperparameter
    tuner's JSON output via `CheckerConfig.from_tuning_json()`.
    """

    # --- Prophet model parameters ---
    changepoint_prior_scale: float = 0.3
    """Controls trend flexibility. Higher = more sensitive to trend changes.
    - 0.05 (default Prophet): too smooth for volatile feeds
    - 0.3 (recommended): good balance for most feeds
    - 0.5-1.0: very aggressive, use for extremely volatile feeds
    """

    changepoint_range: float = 0.95
    """Fraction of training data where changepoints are allowed.
    0.95 lets Prophet detect changes near the end of the training window.
    """

    seasonality_prior_scale: float = 5.0
    """Controls seasonality strength. Higher = stronger weekly patterns."""

    interval_width: float = 0.80
    """Width of the prediction interval (0.80 = 80% CI).
    Increase to 0.90-0.95 to reduce false positives.
    """

    n_changepoints: int = 25
    """Number of potential changepoints in the trend."""

    # --- Growth & Transform ---
    growth_type: str = "logistic"
    """Growth model: 'logistic' (bounded, recommended for count data) or 'linear'."""

    cap_multiplier: float = 3.0
    """Cap = max(y) * cap_multiplier. Only used with logistic growth."""

    log_transform: bool = True
    """Whether to apply log1p transform. Prevents negative predictions."""

    # --- Seasonality ---
    seasonality_mode: str = "multiplicative"
    """'multiplicative' (recommended for count data) or 'additive'."""

    weekly_seasonality_order: int = 3
    """Fourier order for weekly seasonality (2-10). Higher = more flexible."""

    yearly_seasonality_order: int = 0
    """Fourier order for yearly seasonality. 0 = disabled.
    Enable (5-10) if you have > 1 year of data."""

    monthly_fourier_order: int = 0
    """Fourier order for monthly seasonality. 0 = disabled."""

    # --- Holidays ---
    holidays_df: Optional[pd.DataFrame] = field(default=None, repr=False)
    """Custom holiday DataFrame in Prophet format.
    Columns: ['holiday', 'ds', 'lower_window', 'upper_window']
    Use `build_holiday_dataframe()` from dqc_hyperparameter_tuner to create this.
    """

    country_holidays: Optional[str] = None
    """Country code for built-in holidays (e.g., 'US', 'IN', 'GB').
    Use this for quick setup; use holidays_df for custom business calendars.
    """

    holiday_prior_scale: float = 10.0
    """Controls holiday effect strength. Higher = stronger holiday effects."""

    # --- Anomaly detection thresholds ---
    pct_deviation_threshold: float = 0.50
    """Percentage deviation from predicted to flag. 0.50 = 50%."""

    zscore_threshold: float = 2.0
    """Z-score threshold for sudden anomalies. Lower = more sensitive."""

    trend_decline_pct: float = 0.40
    """Flag when rolling average drops this much from baseline. 0.40 = 40% drop."""

    rolling_window: int = 10
    """Number of business days for rolling calculations."""

    min_count_for_alert: int = 10
    """Minimum predicted count to trigger alerts (skip weekends/holidays)."""

    max_training_days: Optional[int] = None
    """If set, only use the most recent N days for training (recency bias)."""

    min_signals_for_warning: int = 2
    """Minimum number of flagged signals to raise a WARNING."""

    min_signals_for_critical: int = 3
    """Minimum number of flagged signals to raise a CRITICAL alert."""

    @classmethod
    def from_tuning_json(
        cls,
        json_path: str,
        use_best: bool = True,
        use_midpoint: bool = False,
    ) -> "CheckerConfig":
        """
        Load configuration from the hyperparameter tuner's JSON output.

        Parameters
        ----------
        json_path : str
            Path to tuning_results_XXX.json from dqc_hyperparameter_tuner.py
        use_best : bool
            If True, use the best values found. If False, use suggested
            range midpoints (safer for production, more robust).
        use_midpoint : bool
            If True, use the midpoint of the suggested ranges instead of
            the best values. Good for production robustness.

        Returns
        -------
        CheckerConfig with parameters from the tuning run.

        Example
        -------
        >>> config = CheckerConfig.from_tuning_json("tuning_results_MT_Incoming_2026.json")
        >>> checker = DQCCountDeviationChecker(config)
        >>> checker.fit_and_predict(df_historical)

        For a second round of fine-tuning, use the ranges:
        >>> with open("tuning_results_XXX.json") as f:
        ...     results = json.load(f)
        >>> ranges = results["suggested_ranges"]
        >>> # Use ranges["changepoint_prior_scale"]["suggested_min"] and
        >>> #     ranges["changepoint_prior_scale"]["suggested_max"]
        >>> # as bounds for your production fine-tuning loop.
        """
        with open(json_path, "r") as f:
            results = json.load(f)

        best_params = results.get("best_params", {})
        suggested_ranges = results.get("suggested_ranges", {})
        model_config = results.get("model_config", {})

        kwargs = {}

        # Helper to get a parameter value
        def _get_value(param_name: str, default=None):
            if use_midpoint and param_name in suggested_ranges:
                info = suggested_ranges[param_name]
                if info.get("type") == "categorical":
                    return info.get("suggested", default)
                smin = info.get("suggested_min", default)
                smax = info.get("suggested_max", default)
                if smin is not None and smax is not None:
                    if info.get("type") == "int":
                        return int(round((smin + smax) / 2))
                    return (smin + smax) / 2
            if use_best and param_name in best_params:
                return best_params[param_name]
            return default

        # Prophet core params
        kwargs["changepoint_prior_scale"] = _get_value(
            "changepoint_prior_scale", cls.changepoint_prior_scale
        )
        kwargs["seasonality_prior_scale"] = _get_value(
            "seasonality_prior_scale", cls.seasonality_prior_scale
        )
        kwargs["changepoint_range"] = _get_value(
            "changepoint_range", cls.changepoint_range
        )
        kwargs["interval_width"] = _get_value(
            "interval_width", cls.interval_width
        )
        kwargs["n_changepoints"] = _get_value(
            "n_changepoints", cls.n_changepoints
        )

        # Seasonality
        mode = _get_value("seasonality_mode", cls.seasonality_mode)
        kwargs["seasonality_mode"] = mode
        kwargs["weekly_seasonality_order"] = _get_value(
            "weekly_seasonality_order", cls.weekly_seasonality_order
        )
        kwargs["yearly_seasonality_order"] = _get_value(
            "yearly_seasonality_order", cls.yearly_seasonality_order
        )
        kwargs["monthly_fourier_order"] = _get_value(
            "monthly_fourier_order", cls.monthly_fourier_order
        )

        # Model pipeline
        kwargs["growth_type"] = model_config.get(
            "growth_type",
            _get_value("growth_type", cls.growth_type),
        )
        kwargs["log_transform"] = model_config.get(
            "log_transform",
            _get_value("log_transform", cls.log_transform),
        )
        kwargs["cap_multiplier"] = _get_value(
            "cap_multiplier", cls.cap_multiplier
        )

        # Holidays
        kwargs["holiday_prior_scale"] = _get_value(
            "holiday_prior_scale", cls.holiday_prior_scale
        )

        logger.info(
            "Loaded CheckerConfig from %s (use_best=%s, use_midpoint=%s)",
            json_path, use_best, use_midpoint,
        )

        return cls(**kwargs)

    def get_suggested_ranges(self, json_path: str) -> dict:
        """
        Load the suggested ranges from a tuning JSON for second-round tuning.

        Returns a dict that you can use to set up a narrow search space:
            {
                "changepoint_prior_scale": {"min": 0.37, "max": 1.0},
                "seasonality_prior_scale": {"min": 0.10, "max": 0.15},
                ...
            }
        """
        with open(json_path, "r") as f:
            results = json.load(f)
        ranges = results.get("suggested_ranges", {})
        simplified = {}
        for param, info in ranges.items():
            if info.get("type") == "categorical":
                simplified[param] = {"suggested": info.get("suggested")}
            else:
                simplified[param] = {
                    "min": info.get("suggested_min"),
                    "max": info.get("suggested_max"),
                }
        return simplified


class DQCCountDeviationChecker:
    """
    Data Quality Control - Count Deviation Checker.

    Uses an improved Prophet model with log-transform, logistic growth,
    and multi-signal anomaly detection to identify count deviations
    in data feeds.

    v2: Now supports configurable growth type, holidays, and tuned
    seasonality orders loaded from the hyperparameter tuner.
    """

    def __init__(self, config: Optional[CheckerConfig] = None):
        self.config = config or CheckerConfig()
        self._model: Optional[Prophet] = None
        self._forecast: Optional[pd.DataFrame] = None
        self._training_cap: float = 0.0

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_training_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Prophet with configurable transform and regressors.

        Applies log1p transform (if enabled) so the model works in log-space,
        preventing negative predictions when inverse-transformed.
        """
        df = df_raw[["ds", "y"]].copy()
        df = df.dropna(subset=["y"])
        df["y"] = df["y"].astype(float)

        # Apply recency window
        if self.config.max_training_days is not None:
            cutoff = df["ds"].max() - pd.Timedelta(
                days=self.config.max_training_days
            )
            df = df[df["ds"] >= cutoff].copy()

        # Transform based on config
        if self.config.log_transform:
            df["y"] = np.log1p(df["y"])
            self._training_cap = float(
                np.log1p(df_raw["y"].max() * self.config.cap_multiplier)
            )
        else:
            self._training_cap = float(
                df_raw["y"].max() * self.config.cap_multiplier
            )

        # Logistic growth needs floor and cap
        if self.config.growth_type == "logistic":
            df["floor"] = 0.0
            df["cap"] = self._training_cap

        # Weekend regressor
        df["is_weekday"] = (df["ds"].dt.dayofweek < 5).astype(float)

        return df

    def _prepare_future(self, future: pd.DataFrame) -> pd.DataFrame:
        """Add floor, cap, and regressors to a future DataFrame."""
        if self.config.growth_type == "logistic":
            future["floor"] = 0.0
            future["cap"] = self._training_cap
        future["is_weekday"] = (future["ds"].dt.dayofweek < 5).astype(float)
        return future

    def _inverse_transform(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """Convert log-space predictions back to original counts."""
        result = forecast.copy()
        if self.config.log_transform:
            for col in ("yhat", "yhat_lower", "yhat_upper"):
                if col in result.columns:
                    result[col] = np.expm1(result[col]).clip(lower=0)
        else:
            for col in ("yhat", "yhat_lower", "yhat_upper"):
                if col in result.columns:
                    result[col] = result[col].clip(lower=0)
        return result

    def _build_holidays(self) -> Optional[pd.DataFrame]:
        """Build the holidays DataFrame from config."""
        frames = []

        if self.config.holidays_df is not None:
            frames.append(self.config.holidays_df)

        # Country holidays are handled via add_country_holidays() on the model
        # so we don't include them here

        if not frames:
            return None

        holidays = pd.concat(frames, ignore_index=True)
        holidays["ds"] = pd.to_datetime(holidays["ds"])
        return holidays

    # ------------------------------------------------------------------
    # Fit & Predict
    # ------------------------------------------------------------------

    def fit_and_predict(
        self, df_historical: pd.DataFrame, forecast_days: int = 7
    ) -> pd.DataFrame:
        """
        Fit Prophet on historical data and generate forecast.

        Parameters
        ----------
        df_historical : DataFrame
            Must contain columns 'ds' (datetime) and 'y' (record count).
        forecast_days : int
            Number of days to forecast beyond the last training date.

        Returns
        -------
        DataFrame
            Forecast in original scale (counts, not log).
        """
        cfg = self.config
        train = self._prepare_training_data(df_historical)

        # Build holidays
        holidays = self._build_holidays()

        self._model = Prophet(
            growth=cfg.growth_type,
            seasonality_mode=cfg.seasonality_mode,
            changepoint_prior_scale=cfg.changepoint_prior_scale,
            changepoint_range=cfg.changepoint_range,
            seasonality_prior_scale=cfg.seasonality_prior_scale,
            n_changepoints=cfg.n_changepoints,
            yearly_seasonality=False,   # Added manually below
            weekly_seasonality=False,   # Added manually below
            daily_seasonality=False,
            interval_width=cfg.interval_width,
            holidays=holidays,
        )

        # Add country holidays if specified
        if cfg.country_holidays:
            try:
                self._model.add_country_holidays(country_name=cfg.country_holidays)
            except Exception as e:
                logger.warning(
                    "Could not add country holidays for '%s': %s",
                    cfg.country_holidays, e,
                )

        # Add weekly seasonality with tuned Fourier order
        self._model.add_seasonality(
            name="weekly",
            period=7,
            fourier_order=cfg.weekly_seasonality_order,
            mode=cfg.seasonality_mode,
        )

        # Add yearly seasonality if enabled
        if cfg.yearly_seasonality_order > 0:
            self._model.add_seasonality(
                name="yearly",
                period=365.25,
                fourier_order=cfg.yearly_seasonality_order,
                mode=cfg.seasonality_mode,
            )

        # Add monthly seasonality if enabled
        if cfg.monthly_fourier_order > 0:
            self._model.add_seasonality(
                name="monthly",
                period=30.5,
                fourier_order=cfg.monthly_fourier_order,
                mode=cfg.seasonality_mode,
            )

        # Add weekday regressor
        self._model.add_regressor("is_weekday", mode=cfg.seasonality_mode)

        logger.info(
            "Fitting Prophet on %d rows (%s to %s), growth=%s, log=%s",
            len(train),
            train["ds"].min().date(),
            train["ds"].max().date(),
            cfg.growth_type,
            cfg.log_transform,
        )
        self._model.fit(train)

        future = self._model.make_future_dataframe(periods=forecast_days)
        future = self._prepare_future(future)

        forecast_raw = self._model.predict(future)
        self._forecast = self._inverse_transform(forecast_raw)

        logger.info(
            "Forecast generated: %d rows, min lower_bound=%.0f",
            len(self._forecast),
            self._forecast["yhat_lower"].min(),
        )
        return self._forecast

    # ------------------------------------------------------------------
    # Single-day check
    # ------------------------------------------------------------------

    def check(
        self,
        today_date,
        today_count: float,
        df_recent_history: pd.DataFrame,
    ) -> dict:
        """
        Check if today's count is anomalous.

        Parameters
        ----------
        today_date : date-like
            The date to check.
        today_count : float
            Actual record count received today.
        df_recent_history : DataFrame
            Recent historical data ('ds', 'y') for rolling calculations.

        Returns
        -------
        dict with keys:
            status : str    — 'OK', 'INFO', 'WARNING', 'CRITICAL', or 'SKIP'
            message : str   — Human-readable summary
            signals : dict  — Detailed per-signal breakdown
        """
        if self._forecast is None:
            raise RuntimeError("Call fit_and_predict() before check()")

        cfg = self.config
        today_ts = pd.Timestamp(today_date)

        # Get Prophet's prediction for today
        pred = self._forecast[self._forecast["ds"] == today_ts]
        if pred.empty:
            return {
                "status": "NO_PREDICTION",
                "message": f"No forecast available for {today_date}",
            }

        yhat = float(pred["yhat"].iloc[0])
        yhat_lower = max(0.0, float(pred["yhat_lower"].iloc[0]))
        yhat_upper = float(pred["yhat_upper"].iloc[0])

        # Skip weekends / low-volume days
        if today_ts.dayofweek >= 5 or yhat < cfg.min_count_for_alert:
            return {
                "status": "SKIP",
                "message": "Weekend/low-volume day — check skipped",
                "predicted": yhat,
                "actual": today_count,
            }

        signals = {}

        # ---- Signal 1: Percentage deviation ----
        pct_dev = (today_count - yhat) / yhat if yhat > 0 else 0.0
        signals["pct_deviation"] = {
            "value": pct_dev,
            "flagged": abs(pct_dev) > cfg.pct_deviation_threshold,
            "direction": "LOW" if pct_dev < 0 else "HIGH",
        }

        # ---- Signal 2: Below lower bound (floor-clamped) ----
        signals["below_lower_bound"] = {
            "value": today_count,
            "lower_bound": yhat_lower,
            "flagged": today_count < yhat_lower,
        }

        # ---- Signal 3: Rolling Z-score ----
        biz_history = df_recent_history[
            df_recent_history["y"] > cfg.min_count_for_alert
        ]["y"]
        if len(biz_history) >= 5:
            window = biz_history.tail(cfg.rolling_window)
            mu, sigma = window.mean(), window.std()
            if sigma > 0:
                z = (today_count - mu) / sigma
                signals["zscore"] = {
                    "value": z,
                    "flagged": abs(z) > cfg.zscore_threshold,
                    "direction": "LOW" if z < 0 else "HIGH",
                }

        # ---- Signal 4: Trend decline ----
        if len(biz_history) >= cfg.rolling_window * 3:
            baseline = biz_history.head(cfg.rolling_window).mean()
            recent = biz_history.tail(cfg.rolling_window).mean()
            ratio = recent / baseline if baseline > 0 else 1.0
            signals["trend_decline"] = {
                "value": ratio,
                "flagged": ratio < (1.0 - cfg.trend_decline_pct),
                "baseline": baseline,
                "recent_avg": recent,
            }

        # ---- Aggregate ----
        n_flagged = sum(1 for s in signals.values() if s.get("flagged", False))

        if n_flagged >= cfg.min_signals_for_critical:
            status = "CRITICAL"
        elif n_flagged >= cfg.min_signals_for_warning:
            status = "WARNING"
        elif n_flagged == 1:
            status = "INFO"
        else:
            status = "OK"

        return {
            "status": status,
            "date": str(today_date),
            "actual": today_count,
            "predicted": round(yhat),
            "lower_bound": round(yhat_lower),
            "upper_bound": round(yhat_upper),
            "pct_deviation": round(pct_dev * 100, 1),
            "flagged_signals": n_flagged,
            "total_signals": len(signals),
            "signals": signals,
            "message": (
                f"{status}: {n_flagged}/{len(signals)} signals flagged | "
                f"Actual={today_count} vs Predicted={yhat:.0f} "
                f"({pct_dev*100:+.1f}%)"
            ),
        }

    # ------------------------------------------------------------------
    # Batch backtest
    # ------------------------------------------------------------------

    def backtest(
        self, df_full: pd.DataFrame, start_from_day: int = 30
    ) -> pd.DataFrame:
        """
        Run check() on every day in the historical data (for validation).

        Parameters
        ----------
        df_full : DataFrame
            Full historical data ('ds', 'y').
        start_from_day : int
            Skip the first N days (need history for rolling calcs).

        Returns
        -------
        DataFrame with check results for every day.
        """
        if self._forecast is None:
            self.fit_and_predict(df_full)

        records = []
        for i in range(start_from_day, len(df_full)):
            row = df_full.iloc[i]
            history = df_full.iloc[:i]
            result = self.check(row["ds"], row["y"], history)
            records.append(result)

        return pd.DataFrame(records)


# ======================================================================
# Quick usage example
# ======================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dqc_count_checker.py <csv_path> [tuning_json_path]")
        print("  CSV must have columns: ds, y")
        print("  tuning_json_path: optional, path to tuning results JSON")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    df["ds"] = pd.to_datetime(df["ds"], format="%d-%b-%y")
    df = df.dropna(subset=["y"])
    df["y"] = df["y"].astype(float)

    # Load config from tuning results if provided
    if len(sys.argv) >= 3:
        tuning_json = sys.argv[2]
        print(f"Loading tuned parameters from {tuning_json}")
        config = CheckerConfig.from_tuning_json(tuning_json)
    else:
        # Use aggressive config for volatile feeds
        config = CheckerConfig(
            changepoint_prior_scale=0.3,
            changepoint_range=0.95,
            pct_deviation_threshold=0.50,
            trend_decline_pct=0.40,
            zscore_threshold=2.0,
        )

    checker = DQCCountDeviationChecker(config)
    checker.fit_and_predict(df, forecast_days=7)

    # Backtest
    results = checker.backtest(df)
    print("\n=== Backtest Summary ===")
    print(results["status"].value_counts().to_string())

    # Show warnings and criticals
    alerts = results[results["status"].isin(["WARNING", "CRITICAL"])]
    if not alerts.empty:
        print(f"\n=== {len(alerts)} Alerts ===")
        for _, row in alerts.iterrows():
            print(f"  {row['message']}")
