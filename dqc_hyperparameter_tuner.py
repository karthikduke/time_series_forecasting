"""
DQC Prophet Hyperparameter Tuner v2
====================================
Uses Optuna + Prophet cross-validation to find optimal hyperparameters
AND model-level pipeline parameters for a given feed's data, then outputs
a NARROW suggested range suitable for production retraining.

v2 Enhancements:
  - Expanded search space (~15 parameters including model pipeline params)
  - Holiday calendar support (built-in country + custom business calendars)
  - Convergence diagnostics (automated early stopping + convergence report)
  - Boundary-hitting detection (warns when params hit search limits)
  - Enhanced JSON output for two-round production fine-tuning

Usage:
    python dqc_hyperparameter_tuner.py <csv_path> [--n-trials 200] [--cv-horizon 7]
    python dqc_hyperparameter_tuner.py <csv_path> --country-holidays US
    python dqc_hyperparameter_tuner.py <csv_path> --custom-holidays holidays.csv

Output:
    - Best parameters found
    - Suggested narrow range for each parameter (for prod config)
    - Convergence diagnostics (is N trials enough?)
    - Visualization of parameter importance + convergence
    - Exports results to JSON
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# Suppress noisy logs during optimization
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


# ======================================================================
# Search Space Definition
# ======================================================================

# Original search bounds — used for clamping suggested ranges
SEARCH_BOUNDS = {
    # Prophet core hyperparameters
    "changepoint_prior_scale": (0.001, 5.0),
    "seasonality_prior_scale": (0.01, 50.0),
    "changepoint_range": (0.70, 0.95),
    "interval_width": (0.70, 0.95),
    "n_changepoints": (5, 80),
    "weekly_seasonality_order": (2, 10),
    # Model pipeline parameters
    "cap_multiplier": (1.5, 5.0),
    "yearly_seasonality_order": (0, 10),
    "monthly_fourier_order": (0, 5),
    "holiday_prior_scale": (0.1, 20.0),
    "holiday_window_before": (0, 3),
    "holiday_window_after": (0, 3),
}

# Parameters that are categorical (not numeric)
CATEGORICAL_PARAMS = {"seasonality_mode", "growth_type", "log_transform"}

# Boundary margin for detecting params that hit search limits
BOUNDARY_MARGIN_PCT = 0.05  # 5% of range


# ======================================================================
# Holiday Calendar Support
# ======================================================================

def build_holiday_dataframe(
    country: Optional[str] = None,
    custom_holidays_df: Optional[pd.DataFrame] = None,
    year_range: tuple = (2024, 2027),
    default_window_before: int = 0,
    default_window_after: int = 1,
) -> Optional[pd.DataFrame]:
    """
    Build a Prophet-compatible holiday DataFrame.

    Supports two sources:
      1. Built-in country holidays (via Prophet's internal calendar)
      2. Custom business-line holidays (your own calendar table)

    Parameters
    ----------
    country : str, optional
        Country code for built-in holidays (e.g., "US", "IN", "GB").
        If None, no country holidays are added.
    custom_holidays_df : DataFrame, optional
        Your custom business calendar. Expected columns:
          - 'holiday_name' or 'holiday': name of the event
          - 'date' or 'ds': date of the event
          - 'lower_window' (optional): days before to include (default: 0)
          - 'upper_window' (optional): days after to include (default: 1)
    year_range : tuple
        (start_year, end_year) for generating country holidays.
    default_window_before : int
        Default lower_window for custom holidays.
    default_window_after : int
        Default upper_window for custom holidays.

    Returns
    -------
    DataFrame or None
        Prophet-format holidays DataFrame with columns:
        ['holiday', 'ds', 'lower_window', 'upper_window']

    Example
    -------
    # Your business calendar might look like:
    # | holiday_name       | date       |
    # |--------------------|------------|
    # | Q1 Close           | 2026-03-31 |
    # | System Maintenance | 2026-04-15 |
    # | Annual Audit       | 2026-06-01 |
    #
    # This function converts it to Prophet's format automatically.
    """
    frames = []

    # --- Custom holidays ---
    if custom_holidays_df is not None:
        df_custom = convert_business_calendar(
            custom_holidays_df,
            default_window_before=default_window_before,
            default_window_after=default_window_after,
        )
        if df_custom is not None and len(df_custom) > 0:
            frames.append(df_custom)
            logger.info("Added %d custom holiday entries", len(df_custom))

    # --- Country holidays (via `holidays` package) ---
    if country is not None:
        try:
            import holidays as holidays_pkg
            country_cls = holidays_pkg.country_holidays(country)
            holidays_list = []
            for year in range(year_range[0], year_range[1] + 1):
                for dt, name in sorted(
                    holidays_pkg.country_holidays(
                        country, years=year
                    ).items()
                ):
                    holidays_list.append({
                        "holiday": name,
                        "ds": pd.Timestamp(dt),
                        "lower_window": -1,
                        "upper_window": 1,
                    })
            if holidays_list:
                country_holidays_df = pd.DataFrame(holidays_list)
                frames.append(country_holidays_df)
                logger.info(
                    "Added %d %s country holiday entries (via holidays pkg)",
                    len(country_holidays_df),
                    country,
                )
        except ImportError:
            logger.info(
                "'holidays' package not installed. "
                "Falling back to manual US holidays."
            )
            # Fallback: manually define major US holidays
            if country.upper() == "US":
                manual = _get_manual_us_holidays(year_range)
                if manual is not None:
                    frames.append(manual)
                    logger.info("Added %d manual US holiday entries", len(manual))

    if not frames:
        return None

    holidays = pd.concat(frames, ignore_index=True)
    holidays["ds"] = pd.to_datetime(holidays["ds"])
    holidays = holidays.drop_duplicates(subset=["holiday", "ds"])
    return holidays


def _get_manual_us_holidays(year_range: tuple) -> pd.DataFrame:
    """Manually define major US holidays as a fallback."""
    holidays_list = []
    for year in range(year_range[0], year_range[1] + 1):
        us_holidays = {
            "New Year's Day": f"{year}-01-01",
            "MLK Day": f"{year}-01-20",
            "Presidents Day": f"{year}-02-17",
            "Memorial Day": f"{year}-05-26",
            "Independence Day": f"{year}-07-04",
            "Labor Day": f"{year}-09-01",
            "Thanksgiving": f"{year}-11-27",
            "Christmas": f"{year}-12-25",
        }
        for name, date_str in us_holidays.items():
            holidays_list.append({
                "holiday": name,
                "ds": pd.Timestamp(date_str),
                "lower_window": -1,
                "upper_window": 1,
            })
    return pd.DataFrame(holidays_list)


def convert_business_calendar(
    business_cal_df: pd.DataFrame,
    default_window_before: int = 0,
    default_window_after: int = 1,
) -> Optional[pd.DataFrame]:
    """
    Convert a business-line-specific calendar to Prophet's holiday format.

    Your business calendar table might have columns like:
        - holiday_name / holiday / event_name / name
        - date / ds / event_date

    This function auto-detects the column names and converts them.

    Parameters
    ----------
    business_cal_df : DataFrame
        Your business calendar with at least a name and date column.
    default_window_before : int
        Days before the holiday to include in the effect.
    default_window_after : int
        Days after the holiday to include in the effect.

    Returns
    -------
    DataFrame with columns: ['holiday', 'ds', 'lower_window', 'upper_window']
    """
    if business_cal_df is None or len(business_cal_df) == 0:
        return None

    df = business_cal_df.copy()

    # Auto-detect the holiday name column
    name_candidates = ["holiday_name", "holiday", "event_name", "name", "event"]
    name_col = None
    for col in name_candidates:
        if col in df.columns:
            name_col = col
            break
    if name_col is None:
        raise ValueError(
            f"Cannot find holiday name column. Expected one of: {name_candidates}. "
            f"Got columns: {list(df.columns)}"
        )

    # Auto-detect the date column
    date_candidates = ["date", "ds", "event_date", "holiday_date"]
    date_col = None
    for col in date_candidates:
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        raise ValueError(
            f"Cannot find date column. Expected one of: {date_candidates}. "
            f"Got columns: {list(df.columns)}"
        )

    # Build the Prophet-format DataFrame
    result = pd.DataFrame({
        "holiday": df[name_col].astype(str),
        "ds": pd.to_datetime(df[date_col]),
        "lower_window": df.get(
            "lower_window",
            pd.Series([-default_window_before] * len(df))
        ).astype(int),
        "upper_window": df.get(
            "upper_window",
            pd.Series([default_window_after] * len(df))
        ).astype(int),
    })

    # Ensure lower_window is negative (days before)
    result["lower_window"] = -result["lower_window"].abs()

    return result


# ======================================================================
# Data preparation
# ======================================================================

def load_and_prepare(
    csv_path: str,
    apply_log_transform: bool = True,
    cap_multiplier: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CSV and return both raw and (optionally) log-transformed DataFrames.

    Parameters
    ----------
    csv_path : str
        Path to CSV with columns: ds, y
    apply_log_transform : bool
        Whether to apply log1p transform.
    cap_multiplier : float
        Multiplier for the cap (max_y * cap_multiplier).
    """
    df_raw = pd.read_csv(csv_path)
    df_raw["ds"] = pd.to_datetime(df_raw["ds"], format="%d-%b-%y")
    df_raw = df_raw.dropna(subset=["y"])
    df_raw["y"] = df_raw["y"].astype(float)

    df_prepared = df_raw[["ds", "y"]].copy()
    df_prepared["y_original"] = df_prepared["y"].copy()

    if apply_log_transform:
        df_prepared["y"] = np.log1p(df_prepared["y"])
        df_prepared["cap"] = float(np.log1p(df_raw["y"].max() * cap_multiplier))
    else:
        df_prepared["cap"] = float(df_raw["y"].max() * cap_multiplier)

    df_prepared["floor"] = 0.0
    df_prepared["is_weekday"] = (df_prepared["ds"].dt.dayofweek < 5).astype(float)

    return df_raw, df_prepared


# ======================================================================
# Convergence Diagnostics
# ======================================================================

class ConvergenceTracker:
    """
    Optuna callback that tracks convergence and supports early stopping.

    This answers the key question: "Were N trials enough?"

    How it works:
      - After each trial, checks if the best value improved
      - If no improvement for `patience` consecutive trials, stops the study
      - Records a full convergence history for post-hoc analysis

    How to read convergence:
      - If `is_converged = True`: the best value plateaued. Your trial budget
        was sufficient. The suggested ranges are reliable.
      - If `is_converged = False`: the optimizer was still finding improvements
        when it stopped. Consider running more trials or narrowing the search.
    """

    def __init__(
        self,
        patience: int = 30,
        min_delta: float = 0.001,
        n_trials: int = 200,
    ):
        """
        Parameters
        ----------
        patience : int
            Stop if no improvement for this many consecutive trials.
            Rule of thumb: 2x the number of parameters you're tuning.
        min_delta : float
            Minimum improvement to count as "progress".
            For RMSE in log-space, 0.001 is a reasonable threshold.
        n_trials : int
            Total trials requested (for progress logging).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.n_trials = n_trials

        # State tracking
        self.best_value: Optional[float] = None
        self.best_trial_number: int = 0
        self.trials_since_improvement: int = 0
        self.is_converged: bool = False
        self.convergence_history: list[dict] = []
        self.start_time: float = time.time()

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Called after each trial completes."""
        current_best = study.best_value
        elapsed = time.time() - self.start_time

        # Track convergence
        improved = False
        if self.best_value is None or current_best < self.best_value - self.min_delta:
            self.best_value = current_best
            self.best_trial_number = trial.number
            self.trials_since_improvement = 0
            improved = True
        else:
            self.trials_since_improvement += 1

        # Record history
        self.convergence_history.append({
            "trial": trial.number,
            "value": trial.value if trial.value != float("inf") else None,
            "best_value": current_best,
            "improved": improved,
            "elapsed_seconds": round(elapsed, 1),
        })

        # Progress logging
        if (trial.number + 1) % 10 == 0 or trial.number == 0:
            logger.info(
                "Trial %3d/%d | Best RMSE: %.6f (trial #%d) | "
                "No improvement for: %d trials | Elapsed: %.0fs",
                trial.number + 1,
                self.n_trials,
                current_best,
                self.best_trial_number,
                self.trials_since_improvement,
                elapsed,
            )

        # Early stopping check
        if self.trials_since_improvement >= self.patience:
            self.is_converged = True
            logger.info(
                "[OK] CONVERGED: No improvement for %d trials. "
                "Stopping at trial %d (best was trial #%d, RMSE=%.6f)",
                self.patience,
                trial.number + 1,
                self.best_trial_number,
                current_best,
            )
            study.stop()


def compute_convergence_report(
    study: optuna.Study,
    tracker: ConvergenceTracker,
) -> dict:
    """
    Analyze the optimization run and produce a convergence diagnostic.

    This is the KEY function for answering "Were my trials enough?"

    Returns a dict with:
      - is_converged: Did the best value plateau?
      - plateau_at_trial: Which trial the best value stabilized
      - trials_since_improvement: How many trials ran without improvement
      - improvement_rate_early_vs_late: Ratio of improvement rates
      - boundary_warnings: Params that hit search bounds (need wider search)
      - recommendation: Human-readable suggestion

    How to Interpret:
    -----------------
    1. is_converged = True, no boundary_warnings:
       → Perfect. Your ranges are reliable. Use them as-is.

    2. is_converged = True, WITH boundary_warnings:
       → Converged but some params maxed out. Widen those specific bounds
         and re-run to confirm the boundary wasn't limiting quality.

    3. is_converged = False:
       → Still improving when stopped. Run more trials.
         Check improvement_rate_early_vs_late:
         - If ratio > 0.5: still making significant progress → double trials
         - If ratio < 0.1: nearly converged → add 50 more trials
    """
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value != float("inf")
    ]
    if len(completed) < 10:
        return {
            "is_converged": False,
            "recommendation": "Too few completed trials for convergence analysis. "
                              "Run at least 20 trials.",
            "n_completed": len(completed),
        }

    completed.sort(key=lambda t: t.number)
    values = [t.value for t in completed]

    # --- Compute improvement rates: early vs late ---
    quarter = max(5, len(values) // 4)
    early_values = values[:quarter]
    late_values = values[-quarter:]

    early_improvement = max(early_values) - min(early_values)
    late_improvement = max(late_values) - min(late_values)

    improvement_ratio = (
        late_improvement / early_improvement
        if early_improvement > 0 else 0.0
    )

    # --- Detect boundary-hitting parameters ---
    boundary_warnings = _detect_boundary_hits(study)

    # --- Build best-value trajectory ---
    best_so_far = []
    current_best = float("inf")
    for v in values:
        current_best = min(current_best, v)
        best_so_far.append(current_best)

    # Find plateau point (last trial that improved the best)
    plateau_at = 0
    for i in range(1, len(best_so_far)):
        if best_so_far[i] < best_so_far[i - 1] - tracker.min_delta:
            plateau_at = i

    trials_after_plateau = len(values) - 1 - plateau_at

    # --- Build recommendation ---
    if tracker.is_converged and not boundary_warnings:
        recommendation = (
            f"[OK] CONVERGED -- Best RMSE stable for {tracker.trials_since_improvement} "
            f"trials after trial #{tracker.best_trial_number}. "
            f"Your suggested ranges are reliable for production use."
        )
    elif tracker.is_converged and boundary_warnings:
        params_str = ", ".join(boundary_warnings)
        recommendation = (
            f"[WARN] CONVERGED but {len(boundary_warnings)} parameter(s) hit search bounds: "
            f"[{params_str}]. Consider widening those bounds and re-running "
            f"to confirm the boundary isn't limiting model quality."
        )
    elif improvement_ratio < 0.1:
        recommendation = (
            f"Nearly converged (improvement rate dropped to {improvement_ratio:.1%} "
            f"of early rate). Running 30-50 more trials should confirm convergence."
        )
    elif improvement_ratio < 0.5:
        recommendation = (
            f"Still improving moderately (late improvement = {improvement_ratio:.1%} "
            f"of early rate). Consider running 50-100 more trials."
        )
    else:
        recommendation = (
            f"[FAIL] NOT CONVERGED -- Still actively improving "
            f"(late improvement = {improvement_ratio:.1%} of early rate). "
            f"Recommend doubling the trial count."
        )

    return {
        "is_converged": tracker.is_converged,
        "plateau_at_trial": plateau_at,
        "best_trial_number": tracker.best_trial_number,
        "trials_since_improvement": tracker.trials_since_improvement,
        "total_completed_trials": len(completed),
        "improvement_rate_early_vs_late": round(improvement_ratio, 4),
        "early_improvement": round(early_improvement, 6),
        "late_improvement": round(late_improvement, 6),
        "boundary_warnings": boundary_warnings,
        "recommendation": recommendation,
    }


def _detect_boundary_hits(study: optuna.Study) -> list[str]:
    """
    Check if the best trial's parameters are near search bounds.

    If a param is within 5% of its search bound, it means the optimizer
    "wanted" to go further but was stopped by the bound. You should
    widen the search space for that parameter.
    """
    warnings = []
    best_params = study.best_params

    for param_name, best_value in best_params.items():
        if param_name in CATEGORICAL_PARAMS:
            continue
        if param_name not in SEARCH_BOUNDS:
            continue

        lo, hi = SEARCH_BOUNDS[param_name]
        span = hi - lo
        margin = span * BOUNDARY_MARGIN_PCT

        if best_value <= lo + margin:
            warnings.append(f"{param_name} near LOWER bound ({best_value:.4f} ~ {lo})")
        elif best_value >= hi - margin:
            warnings.append(f"{param_name} near UPPER bound ({best_value:.4f} ~ {hi})")

    return warnings


# ======================================================================
# Optuna objective function
# ======================================================================

def create_objective(
    df_raw: pd.DataFrame,
    cv_initial: str,
    cv_period: str,
    cv_horizon: str,
    holidays_df: Optional[pd.DataFrame] = None,
):
    """
    Returns an Optuna objective function that fits Prophet with sampled
    hyperparameters (including model pipeline params) and evaluates via
    cross-validation RMSE.
    """

    def objective(trial: optuna.Trial) -> float:
        # === PHASE 1: Model Pipeline Parameters ===
        # These control HOW data is prepared before Prophet sees it.

        growth_type = trial.suggest_categorical(
            "growth_type", ["logistic", "linear"]
        )
        log_transform = trial.suggest_categorical(
            "log_transform", [True, False]
        )
        cap_multiplier = trial.suggest_float(
            "cap_multiplier", 1.5, 5.0, step=0.5
        )

        # === PHASE 2: Prophet Core Hyperparameters ===
        # These tune Prophet's internal model behavior.

        changepoint_prior_scale = trial.suggest_float(
            "changepoint_prior_scale", 0.001, 5.0, log=True
        )
        seasonality_prior_scale = trial.suggest_float(
            "seasonality_prior_scale", 0.01, 50.0, log=True
        )
        changepoint_range = trial.suggest_float(
            "changepoint_range", 0.70, 0.95, step=0.05
        )
        seasonality_mode = trial.suggest_categorical(
            "seasonality_mode", ["additive", "multiplicative"]
        )
        interval_width = trial.suggest_float(
            "interval_width", 0.70, 0.95, step=0.05
        )
        n_changepoints = trial.suggest_int(
            "n_changepoints", 5, 80, step=5
        )
        weekly_seasonality_order = trial.suggest_int(
            "weekly_seasonality_order", 2, 10
        )

        # === PHASE 3: Seasonality Parameters ===

        yearly_seasonality_order = trial.suggest_int(
            "yearly_seasonality_order", 0, 10
        )
        monthly_fourier_order = trial.suggest_int(
            "monthly_fourier_order", 0, 5
        )

        # === PHASE 4: Holiday Parameters ===
        if holidays_df is not None:
            holiday_prior_scale = trial.suggest_float(
                "holiday_prior_scale", 0.1, 20.0, log=True
            )
            holiday_window_before = trial.suggest_int(
                "holiday_window_before", 0, 3
            )
            holiday_window_after = trial.suggest_int(
                "holiday_window_after", 0, 3
            )
        else:
            holiday_prior_scale = 10.0
            holiday_window_before = 0
            holiday_window_after = 1

        try:
            # --- Prepare data based on pipeline params ---
            df_prepared = df_raw[["ds", "y"]].copy()
            df_prepared["y_original"] = df_prepared["y"].copy()

            if log_transform:
                df_prepared["y"] = np.log1p(df_prepared["y"])
                df_prepared["cap"] = float(
                    np.log1p(df_raw["y"].max() * cap_multiplier)
                )
            else:
                df_prepared["cap"] = float(
                    df_raw["y"].max() * cap_multiplier
                )

            df_prepared["floor"] = 0.0
            df_prepared["is_weekday"] = (
                df_prepared["ds"].dt.dayofweek < 5
            ).astype(float)

            # --- Prepare holidays with tuned windows ---
            model_holidays = None
            if holidays_df is not None:
                model_holidays = holidays_df.copy()
                model_holidays["lower_window"] = -abs(holiday_window_before)
                model_holidays["upper_window"] = holiday_window_after
                model_holidays["prior_scale"] = holiday_prior_scale

            # --- Build Prophet model ---
            m = Prophet(
                growth=growth_type,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                changepoint_range=changepoint_range,
                seasonality_mode=seasonality_mode,
                interval_width=interval_width,
                n_changepoints=n_changepoints,
                weekly_seasonality=False,   # Added manually with tuned order
                yearly_seasonality=False,   # Added manually if order > 0
                daily_seasonality=False,
                holidays=model_holidays,
            )

            # Add weekly seasonality with tuned Fourier order
            m.add_seasonality(
                name="weekly",
                period=7,
                fourier_order=weekly_seasonality_order,
                mode=seasonality_mode,
            )

            # Add yearly seasonality if order > 0
            if yearly_seasonality_order > 0:
                m.add_seasonality(
                    name="yearly",
                    period=365.25,
                    fourier_order=yearly_seasonality_order,
                    mode=seasonality_mode,
                )

            # Add monthly seasonality if order > 0
            if monthly_fourier_order > 0:
                m.add_seasonality(
                    name="monthly",
                    period=30.5,
                    fourier_order=monthly_fourier_order,
                    mode=seasonality_mode,
                )

            # Add weekday regressor
            m.add_regressor("is_weekday", mode=seasonality_mode)

            # Prepare training data
            train_cols = ["ds", "y", "is_weekday"]
            if growth_type == "logistic":
                train_cols += ["floor", "cap"]
            train = df_prepared[train_cols].copy()

            # Suppress cmdstanpy noise during fit
            import logging as _logging
            _logging.getLogger("cmdstanpy").setLevel(_logging.ERROR)

            m.fit(train)

            # --- Cross-validation ---
            df_cv = cross_validation(
                m,
                initial=cv_initial,
                period=cv_period,
                horizon=cv_horizon,
            )

            # Compute RMSE manually — more robust than performance_metrics
            errors = df_cv["y"] - df_cv["yhat"]
            valid_mask = np.isfinite(errors)
            if valid_mask.sum() == 0:
                logger.warning(
                    "Trial %d: no valid predictions in CV", trial.number
                )
                return float("inf")

            valid_errors = errors[valid_mask]
            rmse = float(np.sqrt((valid_errors ** 2).mean()))
            mae = float(valid_errors.abs().mean())

            if not np.isfinite(rmse):
                logger.warning("Trial %d: RMSE is not finite", trial.number)
                return float("inf")

            # Store additional metrics for later analysis
            trial.set_user_attr("mae", mae)
            trial.set_user_attr("rmse", rmse)
            trial.set_user_attr("n_cv_points", int(valid_mask.sum()))

            # Compute coverage if possible
            try:
                in_bounds = (
                    (df_cv["y"] >= df_cv["yhat_lower"]) &
                    (df_cv["y"] <= df_cv["yhat_upper"])
                )
                trial.set_user_attr("coverage", float(in_bounds.mean()))
            except Exception:
                pass

            return rmse

        except Exception as e:
            logger.warning(
                "Trial %d failed: %s: %s",
                trial.number, type(e).__name__, e,
            )
            return float("inf")

    return objective


# ======================================================================
# Compute narrow ranges from top-K trials
# ======================================================================

def compute_suggested_ranges(
    study: optuna.Study,
    top_k: int = 10,
    margin_pct: float = 0.15,
) -> dict:
    """
    Look at the top-K trials and compute a narrow range for each parameter.

    For each parameter:
      - range = [min(top_k_values), max(top_k_values)]
      - expand by `margin_pct` on each side for production safety
      - clamp to the original search bounds

    Returns a dict of {param_name: {best, min, max, suggested_min, suggested_max, type}}.
    """
    # Get top K completed trials sorted by value
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value != float("inf")
    ]
    completed.sort(key=lambda t: t.value)
    top_trials = completed[:top_k]

    if not top_trials:
        return {}

    best_params = study.best_params

    ranges = {}

    for param_name, best_value in best_params.items():
        if param_name in CATEGORICAL_PARAMS:
            # For categoricals, report the distribution among top-K
            values = [t.params[param_name] for t in top_trials]
            counts = Counter(values)
            ranges[param_name] = {
                "type": "categorical",
                "best": best_value,
                "top_k_distribution": dict(counts),
                "suggested": counts.most_common(1)[0][0],
            }
            continue

        # Numeric parameter
        values = [
            t.params[param_name]
            for t in top_trials
            if param_name in t.params
        ]
        if not values:
            continue

        val_min = min(values)
        val_max = max(values)

        # Expand by margin
        span = val_max - val_min
        if span == 0:
            # All top-K converged to same value — use ±margin_pct of the value
            span = abs(best_value) * margin_pct * 2

        expanded_min = val_min - span * margin_pct
        expanded_max = val_max + span * margin_pct

        # Clamp to original search bounds
        lo_bound, hi_bound = SEARCH_BOUNDS.get(
            param_name, (expanded_min, expanded_max)
        )
        suggested_min = max(expanded_min, lo_bound)
        suggested_max = min(expanded_max, hi_bound)

        # For integer params, round
        is_int = isinstance(best_value, int)
        if is_int:
            suggested_min = int(np.floor(suggested_min))
            suggested_max = int(np.ceil(suggested_max))

        ranges[param_name] = {
            "type": "int" if is_int else "float",
            "best": best_value,
            "top_k_min": round(val_min, 6) if not is_int else int(val_min),
            "top_k_max": round(val_max, 6) if not is_int else int(val_max),
            "suggested_min": (
                round(suggested_min, 6) if not is_int else suggested_min
            ),
            "suggested_max": (
                round(suggested_max, 6) if not is_int else suggested_max
            ),
        }

    return ranges


# ======================================================================
# Pretty-print results
# ======================================================================

def print_results(
    study: optuna.Study,
    ranges: dict,
    convergence: dict,
    df_raw: pd.DataFrame,
    holidays_used: bool = False,
):
    """Print a formatted summary of the tuning results."""

    best = study.best_trial

    print("\n" + "=" * 80)
    print("  OPTUNA HYPERPARAMETER TUNING v2 -- RESULTS")
    print("=" * 80)

    print(f"\n  Dataset         : {len(df_raw)} rows, "
          f"{df_raw['ds'].min().strftime('%d-%b-%Y')} to "
          f"{df_raw['ds'].max().strftime('%d-%b-%Y')}")
    print(f"  Total trials    : {len(study.trials)}")
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    print(f"  Completed       : {len(completed)}")
    print(f"  Best RMSE (log) : {best.value:.6f}")
    if "mae" in best.user_attrs:
        print(f"  Best MAE  (log) : {best.user_attrs['mae']:.6f}")
    if "coverage" in best.user_attrs:
        print(f"  Best Coverage   : {best.user_attrs['coverage']:.2%}")
    print(f"  Holidays used   : {'Yes' if holidays_used else 'No'}")

    # --- Convergence ---
    print("\n" + "-" * 80)
    print("  CONVERGENCE DIAGNOSTICS")
    print("-" * 80)
    print(f"  {convergence['recommendation']}")
    if convergence.get("boundary_warnings"):
        print("  ⚠ Boundary warnings:")
        for w in convergence["boundary_warnings"]:
            print(f"    - {w}")

    # --- Best Parameters (grouped by category) ---
    print("\n" + "-" * 80)
    print("  BEST PARAMETERS FOUND")
    print("-" * 80)

    pipeline_params = ["growth_type", "log_transform", "cap_multiplier"]
    core_params = [
        "changepoint_prior_scale", "seasonality_prior_scale",
        "changepoint_range", "seasonality_mode", "interval_width",
        "n_changepoints", "weekly_seasonality_order",
    ]
    seasonality_params = ["yearly_seasonality_order", "monthly_fourier_order"]
    holiday_params = [
        "holiday_prior_scale", "holiday_window_before", "holiday_window_after",
    ]

    for group_name, param_list in [
        ("Model Pipeline", pipeline_params),
        ("Prophet Core", core_params),
        ("Seasonality", seasonality_params),
        ("Holidays", holiday_params),
    ]:
        group_vals = {k: v for k, v in best.params.items() if k in param_list}
        if not group_vals:
            continue
        print(f"\n  [{group_name}]")
        for k, v in sorted(group_vals.items()):
            if isinstance(v, float):
                print(f"    {k:<30s} = {v:.6f}")
            else:
                print(f"    {k:<30s} = {v}")

    # --- Suggested Ranges ---
    print("\n" + "-" * 80)
    print("  SUGGESTED RANGES FOR PRODUCTION (from top-10 trials + 15% margin)")
    print("-" * 80)
    print(f"  {'Parameter':<30s}  {'Best':>10s}  {'Range Min':>12s}  {'Range Max':>12s}")
    print("  " + "-" * 66)

    for param, info in sorted(ranges.items()):
        if info["type"] == "categorical":
            dist_str = ", ".join(
                f"{k}({v})" for k, v in info["top_k_distribution"].items()
            )
            print(
                f"  {param:<30s}  {str(info['best']):>10s}  "
                f"Suggested: {str(info['suggested']):<20s} [{dist_str}]"
            )
        elif info["type"] == "int":
            print(
                f"  {param:<30s}  {info['best']:>10d}  "
                f"{info['suggested_min']:>12d}  {info['suggested_max']:>12d}"
            )
        else:
            print(
                f"  {param:<30s}  {info['best']:>10.6f}  "
                f"{info['suggested_min']:>12.6f}  {info['suggested_max']:>12.6f}"
            )

    # --- Production Config Snippet ---
    print("\n" + "-" * 80)
    print("  COPY-PASTE CONFIG FOR dqc_count_checker.py")
    print("-" * 80)
    print()
    print("  from dqc_count_checker import CheckerConfig, DQCCountDeviationChecker")
    print()
    print("  # Option 1: Use best values directly")
    print("  config = CheckerConfig(")
    for param in sorted(best.params.keys()):
        val = best.params[param]
        if param in (
            "changepoint_prior_scale", "seasonality_prior_scale",
            "changepoint_range", "interval_width",
        ):
            print(f"      {param}={val:.6f},")
        elif param in ("n_changepoints", "weekly_seasonality_order"):
            print(f"      {param}={val},")
        elif param == "growth_type":
            print(f"      growth_type=\"{val}\",")
    print("  )")
    print()
    print("  # Option 2: Load from JSON (includes ranges for second-round tuning)")
    print("  config = CheckerConfig.from_tuning_json(\"tuning_results_XXX.json\")")
    print()

    # --- Optuna Importance ---
    print("-" * 80)
    print("  PARAMETER IMPORTANCE (fANOVA)")
    print("-" * 80)
    try:
        importances = optuna.importance.get_param_importances(study)
        for param, imp in importances.items():
            bar = "█" * int(imp * 40)
            print(f"    {param:<30s}  {imp:>6.1%}  {bar}")
    except Exception:
        print("    (Could not compute — need more completed trials)")

    print("\n" + "=" * 80)


# ======================================================================
# Visualization
# ======================================================================

def save_visualizations(study: optuna.Study, output_dir: Path):
    """Save Optuna visualization plots if plotly is available."""
    try:
        import plotly

        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_html(str(output_dir / "optuna_optimization_history.html"))

        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_html(str(output_dir / "optuna_param_importance.html"))

        fig3 = optuna.visualization.plot_parallel_coordinate(study)
        fig3.write_html(str(output_dir / "optuna_parallel_coordinate.html"))

        fig4 = optuna.visualization.plot_contour(
            study,
            params=["changepoint_prior_scale", "seasonality_prior_scale"],
        )
        fig4.write_html(str(output_dir / "optuna_contour.html"))

        logger.info("Plotly visualizations saved to %s", output_dir)
    except ImportError:
        logger.info("Install plotly for interactive visualizations: pip install plotly")
    except Exception as e:
        logger.warning("Plotly visualization failed: %s", e)


def save_matplotlib_plots(
    study: optuna.Study,
    ranges: dict,
    convergence: dict,
    tracker: ConvergenceTracker,
    output_dir: Path,
):
    """Generate summary plots with convergence annotations using matplotlib."""
    try:
        import matplotlib.pyplot as plt

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value != float("inf")
        ]
        if not completed:
            return

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # ---- Plot 1: Optimization History + Convergence ----
        ax = axes[0, 0]
        trial_nums = [t.number for t in completed]
        values = [t.value for t in completed]
        ax.scatter(trial_nums, values, alpha=0.3, s=15, color="steelblue", label="Trial RMSE")

        best_vals = [min(values[:i + 1]) for i in range(len(values))]
        ax.plot(trial_nums, best_vals, color="red", linewidth=2, label="Best so far")

        # Convergence annotation
        if convergence.get("plateau_at_trial") is not None:
            plateau_trial = convergence["plateau_at_trial"]
            ax.axvline(
                x=plateau_trial, color="green", linestyle="--", alpha=0.7,
                label=f"Plateau at trial #{plateau_trial}",
            )

        # Early stopping annotation
        if tracker.is_converged:
            ax.annotate(
                f"Converged\n(patience={tracker.patience})",
                xy=(trial_nums[-1], best_vals[-1]),
                xytext=(trial_nums[-1] * 0.7, max(values) * 0.8),
                arrowprops=dict(arrowstyle="->", color="green"),
                fontsize=9, color="green", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5),
            )

        ax.set_xlabel("Trial")
        ax.set_ylabel("RMSE (log-space)")
        ax.set_title("Optimization History + Convergence", fontweight="bold")
        ax.legend(fontsize=8)

        # ---- Plot 2: Parameter importance ----
        ax = axes[0, 1]
        try:
            importances = optuna.importance.get_param_importances(study)
            params_sorted = sorted(
                importances.items(), key=lambda x: x[1], reverse=True
            )
            names = [p[0].replace("_", "\n") for p in params_sorted[:10]]
            imps = [p[1] for p in params_sorted[:10]]
            bars = ax.barh(names, imps, color="teal", alpha=0.7)
            ax.set_xlabel("Importance")
            ax.set_title("Parameter Importance (fANOVA)", fontweight="bold")
            ax.invert_yaxis()
        except Exception:
            ax.text(
                0.5, 0.5, "Not enough trials",
                ha="center", va="center", transform=ax.transAxes,
            )

        # ---- Plot 3: Suggested ranges ----
        ax = axes[1, 0]
        numeric_params = {
            k: v for k, v in ranges.items() if v["type"] != "categorical"
        }
        if numeric_params:
            y_positions = list(range(len(numeric_params)))
            param_names = []
            for i, (param, info) in enumerate(sorted(numeric_params.items())):
                param_names.append(param.replace("_", "\n"))
                lo = info["suggested_min"]
                hi = info["suggested_max"]
                best_val = info["best"]
                ax.barh(
                    i, hi - lo, left=lo, height=0.5,
                    color="lightblue", edgecolor="steelblue", alpha=0.8,
                )
                ax.plot(best_val, i, "D", color="red", markersize=8, zorder=5)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(param_names, fontsize=7)
            ax.set_title(
                "Suggested Ranges (blue) + Best (red ◆)", fontweight="bold"
            )
            ax.invert_yaxis()

        # ---- Plot 4: Top-10 trials table ----
        ax = axes[1, 1]
        ax.axis("off")
        top_10 = sorted(completed, key=lambda t: t.value)[:10]
        table_data = []
        for t in top_10:
            table_data.append([
                t.number,
                f"{t.value:.4f}",
                f"{t.params.get('changepoint_prior_scale', 0):.4f}",
                f"{t.params.get('seasonality_prior_scale', 0):.2f}",
                t.params.get("seasonality_mode", "?"),
                t.params.get("n_changepoints", "?"),
                str(t.params.get("growth_type", "?"))[:3],
            ])
        table = ax.table(
            cellText=table_data,
            colLabels=["Trial", "RMSE", "CPS", "SPS", "Mode", "N_CP", "Grow"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.4)
        ax.set_title("Top-10 Trials", fontweight="bold", pad=20)

        plt.suptitle(
            "Prophet Hyperparameter Tuning v2 — Results",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        plot_path = output_dir / "tuning_results.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Plot saved to %s", plot_path)

    except Exception as e:
        logger.warning("Matplotlib plotting failed: %s", e)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tune Prophet hyperparameters for DQC count deviation (v2)"
    )
    parser.add_argument("csv_path", help="Path to CSV with columns: ds, y")
    parser.add_argument(
        "--n-trials", type=int, default=200,
        help="Max number of Optuna trials (default: 200, early stopping may end sooner)",
    )
    parser.add_argument(
        "--patience", type=int, default=30,
        help="Early stopping patience: stop if no improvement for N trials (default: 30)",
    )
    parser.add_argument(
        "--cv-initial", type=str, default=None,
        help="CV initial training period, e.g. '60 days' (auto-calculated if omitted)",
    )
    parser.add_argument(
        "--cv-period", type=str, default=None,
        help="CV period between cutoffs, e.g. '14 days' (auto-calculated if omitted)",
    )
    parser.add_argument(
        "--cv-horizon", type=str, default="7 days",
        help="CV forecast horizon (default: '7 days')",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of top trials to derive ranges from (default: 10)",
    )
    parser.add_argument(
        "--margin-pct", type=float, default=0.15,
        help="Margin to add around top-K range (default: 0.15 = 15%%)",
    )
    parser.add_argument(
        "--country-holidays", type=str, default=None,
        help="Country code for built-in holidays, e.g. 'US', 'IN', 'GB'",
    )
    parser.add_argument(
        "--custom-holidays", type=str, default=None,
        help="Path to CSV with custom holidays (columns: holiday_name, date)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save results (default: same as CSV)",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    logger.info("Loading data from %s", csv_path)
    df_raw, _ = load_and_prepare(str(csv_path))
    logger.info(
        "Loaded %d rows (%s to %s)",
        len(df_raw),
        df_raw["ds"].min().strftime("%d-%b-%Y"),
        df_raw["ds"].max().strftime("%d-%b-%Y"),
    )

    # --- Build holiday calendar ---
    holidays_df = None
    custom_holidays_df = None
    if args.custom_holidays:
        custom_holidays_df = pd.read_csv(args.custom_holidays)
        logger.info("Loaded custom holidays from %s", args.custom_holidays)

    if args.country_holidays or custom_holidays_df is not None:
        data_years = (
            df_raw["ds"].min().year - 1,
            df_raw["ds"].max().year + 1,
        )
        holidays_df = build_holiday_dataframe(
            country=args.country_holidays,
            custom_holidays_df=custom_holidays_df,
            year_range=data_years,
        )
        if holidays_df is not None:
            logger.info(
                "Holiday calendar: %d entries (%d unique holidays)",
                len(holidays_df),
                holidays_df["holiday"].nunique(),
            )

    # --- Auto-calculate CV parameters based on data size ---
    total_days = (df_raw["ds"].max() - df_raw["ds"].min()).days
    if args.cv_initial is None:
        initial_days = max(30, int(total_days * 0.5))
        cv_initial = f"{initial_days} days"
    else:
        cv_initial = args.cv_initial

    if args.cv_period is None:
        period_days = max(7, int(total_days * 0.1))
        cv_period = f"{period_days} days"
    else:
        cv_period = args.cv_period

    cv_horizon = args.cv_horizon
    logger.info(
        "CV config: initial=%s, period=%s, horizon=%s",
        cv_initial, cv_period, cv_horizon,
    )

    # --- Run Optuna ---
    logger.info(
        "Starting Optuna optimization with up to %d trials "
        "(early stopping patience=%d)...",
        args.n_trials, args.patience,
    )
    start_time = time.time()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        study_name=f"dqc_prophet_v2_{csv_path.stem}",
    )

    objective = create_objective(
        df_raw, cv_initial, cv_period, cv_horizon, holidays_df
    )

    # Convergence tracker (also serves as progress callback)
    tracker = ConvergenceTracker(
        patience=args.patience,
        min_delta=0.001,
        n_trials=args.n_trials,
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        callbacks=[tracker],
    )

    elapsed = time.time() - start_time
    logger.info("Optimization complete in %.1f seconds", elapsed)

    # --- Convergence diagnostics ---
    convergence = compute_convergence_report(study, tracker)

    # --- Compute ranges ---
    ranges = compute_suggested_ranges(
        study, top_k=args.top_k, margin_pct=args.margin_pct
    )

    # --- Print results ---
    print_results(
        study, ranges, convergence, df_raw,
        holidays_used=(holidays_df is not None),
    )

    # --- Save results to JSON ---
    results = {
        "dataset": str(csv_path),
        "total_rows": len(df_raw),
        "date_range": (
            f"{df_raw['ds'].min().isoformat()} to {df_raw['ds'].max().isoformat()}"
        ),
        "n_trials": len(study.trials),
        "n_completed": len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]),
        "best_rmse_log_space": study.best_value,
        "best_params": study.best_params,
        "suggested_ranges": ranges,
        "convergence": convergence,
        "model_config": {
            "growth_type": study.best_params.get("growth_type", "logistic"),
            "log_transform": study.best_params.get("log_transform", True),
            "cap_multiplier": study.best_params.get("cap_multiplier", 3.0),
            "use_holidays": holidays_df is not None,
            "holiday_calendar": (
                (args.country_holidays or "")
                + (" + custom" if custom_holidays_df is not None else "")
            ).strip() or "none",
        },
        "cv_config": {
            "initial": cv_initial,
            "period": cv_period,
            "horizon": cv_horizon,
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    json_path = output_dir / f"tuning_results_{csv_path.stem}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", json_path)

    # --- Save plots ---
    save_matplotlib_plots(study, ranges, convergence, tracker, output_dir)
    save_visualizations(study, output_dir)

    return study, ranges, convergence


if __name__ == "__main__":
    main()
