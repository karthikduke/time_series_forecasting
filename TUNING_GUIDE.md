# DQC Prophet Hyperparameter Tuning Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [How to Know if N Trials is Enough](#how-to-know-if-n-trials-is-enough)
3. [Setting Up Custom Holidays](#setting-up-custom-holidays)
4. [Understanding the JSON Output](#understanding-the-json-output)
5. [Two-Round Tuning Workflow](#two-round-tuning-workflow)
6. [Parameter Reference](#parameter-reference)

---

## Quick Start

### Basic run (no holidays)
```bash
python dqc_hyperparameter_tuner.py MT_Incoming_2026.csv --n-trials 200
```

### With US holidays
```bash
python dqc_hyperparameter_tuner.py MT_Incoming_2026.csv --country-holidays US --n-trials 200
```

### With custom business calendar
```bash
python dqc_hyperparameter_tuner.py MT_Incoming_2026.csv \
    --country-holidays US \
    --custom-holidays my_business_holidays.csv \
    --n-trials 200
```

### Load tuned config into production
```python
from dqc_count_checker import CheckerConfig, DQCCountDeviationChecker

# Use the best values found
config = CheckerConfig.from_tuning_json("tuning_results_MT_Incoming_2026.json")
checker = DQCCountDeviationChecker(config)
checker.fit_and_predict(df_historical)
```

---

## How to Know if N Trials is Enough

This is the most common question when using Optuna. The tuner v2 answers it
automatically via **convergence diagnostics**.

### What the tuner tells you

After every run, check the JSON output's `convergence` section:

```json
{
  "convergence": {
    "is_converged": true,
    "plateau_at_trial": 67,
    "best_trial_number": 67,
    "trials_since_improvement": 33,
    "total_completed_trials": 100,
    "improvement_rate_early_vs_late": 0.02,
    "boundary_warnings": [],
    "recommendation": "✓ CONVERGED — Best RMSE stable for 33 trials..."
  }
}
```

### Decision matrix

| `is_converged` | `boundary_warnings` | `improvement_rate` | What to do |
|:-:|:-:|:-:|:--|
| ✅ True | Empty | < 0.1 | **Done!** Your ranges are reliable. |
| ✅ True | Has warnings | Any | Widen the flagged bounds, re-run |
| ❌ False | Any | > 0.5 | **Double** your trial count |
| ❌ False | Any | 0.1 – 0.5 | Add 50-100 more trials |
| ❌ False | Any | < 0.1 | Nearly there — add 30-50 trials |

### How early stopping works

The tuner uses a **patience-based** early stopping:

```
patience = 30 (default)
```

This means: if the best RMSE hasn't improved by at least `0.001` for 30
consecutive completed trials, the study stops automatically.

**Why 30?** Rule of thumb: patience should be ~2x the number of parameters
you're tuning. With ~15 parameters, patience=30 ensures the TPE sampler
has enough room to explore.

### Reading the optimization plot

The `tuning_results.png` plot shows:
- **Blue dots**: individual trial RMSE values
- **Red line**: best RMSE so far (should plateau if converged)
- **Green dashed line**: where the plateau started
- **Green annotation**: convergence status

If the red line is still going down at the right edge → you need more trials.

### Example: "Did 50 trials work?"

From your previous run with 50 trials:
```
n_changepoints:  best=50, search_max=50  ← HIT BOUNDARY
CPS:             best=0.80, search_max=1.0 ← NEAR BOUNDARY
```

This tells you 50 trials was **not enough** because:
1. Parameters hit the upper bounds → the optimizer wanted to go higher
2. We expanded the bounds in v2 and increased to 200 trials with early stopping

---

## Setting Up Custom Holidays

### Your business calendar format

Your existing calendar table probably looks something like this:

| holiday_name | date | region | business_line |
|:--|:--|:--|:--|
| Q1 Close | 2026-03-31 | US | Finance |
| System Maintenance | 2026-04-15 | ALL | IT |
| Annual Audit | 2026-06-01 | US | Compliance |
| Diwali | 2026-10-20 | IN | ALL |

### Step 1: Create a filtered CSV

Filter your calendar for the specific feed/business line:

```python
import pandas as pd

# Your full business calendar
full_cal = pd.read_sql("SELECT * FROM business_holidays", conn)

# Filter for this feed's region and business line
feed_holidays = full_cal[
    (full_cal["region"].isin(["US", "ALL"])) &
    (full_cal["business_line"].isin(["Finance", "ALL"]))
]

# Save just what the tuner needs
feed_holidays[["holiday_name", "date"]].to_csv(
    "my_business_holidays.csv", index=False
)
```

### Step 2: The tuner auto-converts it

The `convert_business_calendar()` function auto-detects column names:

```python
# It looks for these column name patterns:
# Name:  holiday_name, holiday, event_name, name, event
# Date:  date, ds, event_date, holiday_date

# So any of these work:
#   holiday_name, date         ← recommended
#   holiday, ds                ← Prophet native
#   event_name, event_date     ← also fine
```

### Step 3: Use in tuner + checker

```bash
# Tuning (discovers optimal holiday windows + prior scale)
python dqc_hyperparameter_tuner.py data.csv \
    --country-holidays US \
    --custom-holidays my_business_holidays.csv

# Production (load tuned config)
python dqc_count_checker.py data.csv tuning_results_data.json
```

### Holiday windows (tuned automatically)

The tuner optimizes `holiday_window_before` and `holiday_window_after`:
- `window_before=2`: the effect starts 2 days before the holiday
- `window_after=1`: the effect lasts 1 day after the holiday

This is important because many business events affect volumes for multiple days.

### Combining country + custom holidays

```python
from dqc_hyperparameter_tuner import build_holiday_dataframe
import pandas as pd

custom = pd.read_csv("my_business_holidays.csv")
holidays = build_holiday_dataframe(
    country="US",              # adds US public holidays
    custom_holidays_df=custom,  # adds your business-specific events
    year_range=(2024, 2027),
)
# holidays now contains both US public + your custom events
```

---

## Understanding the JSON Output

### Top-level structure

```json
{
  "dataset": "MT_Incoming_2026.csv",
  "total_rows": 147,
  "n_trials": 100,
  "n_completed": 95,
  "best_rmse_log_space": 1.45,

  "best_params": { ... },         // Exact best values
  "suggested_ranges": { ... },    // Ranges for production
  "convergence": { ... },         // Was N trials enough?
  "model_config": { ... },        // Pipeline decisions
  "cv_config": { ... }            // Cross-validation settings
}
```

### `suggested_ranges` — the key output

Each parameter has a range derived from the **top-10 best trials** plus a 15%
safety margin:

```json
"changepoint_prior_scale": {
  "type": "float",
  "best": 0.801,           // Best single value
  "top_k_min": 0.454,      // Min among top-10 trials
  "top_k_max": 0.995,      // Max among top-10 trials
  "suggested_min": 0.373,  // top_k_min - 15% margin (clamped)
  "suggested_max": 1.0     // top_k_max + 15% margin (clamped)
}
```

**Use `suggested_min` and `suggested_max`** as bounds for your production
fine-tuning. The `best` value is a good starting point within that range.

### `model_config` — pipeline decisions

```json
"model_config": {
  "growth_type": "logistic",    // Use logistic growth
  "log_transform": true,        // Apply log1p to y values
  "cap_multiplier": 3.0,        // Cap = max(y) * 3.0
  "use_holidays": true,
  "holiday_calendar": "US + custom"
}
```

These are **categorical decisions** (not ranges). Use them directly.

---

## Two-Round Tuning Workflow

### Round 1: Wide exploration (this tuner)

```bash
python dqc_hyperparameter_tuner.py data.csv \
    --country-holidays US \
    --n-trials 200
```

This runs a **wide** search across all parameter combinations.

**Output:** `tuning_results_data.json` with suggested ranges.

### Round 2: Narrow production fine-tuning

In your production pipeline, use the ranges for a focused search:

```python
import json
from dqc_count_checker import CheckerConfig

# Load Round 1 ranges
with open("tuning_results_data.json") as f:
    results = json.load(f)
ranges = results["suggested_ranges"]

# Option A: Use best values directly (simplest)
config = CheckerConfig.from_tuning_json("tuning_results_data.json")

# Option B: Use range midpoints (more robust)
config = CheckerConfig.from_tuning_json(
    "tuning_results_data.json",
    use_midpoint=True,
)

# Option C: Run a narrow grid search in production
# (use ranges as bounds for a small grid/random search)
import numpy as np

for _ in range(20):  # Small round-2 search
    cps = np.random.uniform(
        ranges["changepoint_prior_scale"]["suggested_min"],
        ranges["changepoint_prior_scale"]["suggested_max"],
    )
    sps = np.random.uniform(
        ranges["seasonality_prior_scale"]["suggested_min"],
        ranges["seasonality_prior_scale"]["suggested_max"],
    )
    config = CheckerConfig(
        changepoint_prior_scale=cps,
        seasonality_prior_scale=sps,
        # ... other params from ranges
    )
    checker = DQCCountDeviationChecker(config)
    # evaluate on holdout data...
```

### When to re-run Round 1

Re-run the full tuner when:
- You onboard a **new feed** (different data characteristics)
- Your data pattern changes significantly (e.g., seasonal shift)
- The convergence report shows `boundary_warnings`
- Your production monitoring shows degraded accuracy

---

## Parameter Reference

### Model Pipeline Parameters

| Parameter | Type | Range | Purpose |
|:--|:--|:--|:--|
| `growth_type` | categorical | logistic, linear | Growth model. Logistic = bounded (recommended for counts). |
| `log_transform` | categorical | True, False | Apply log1p to y values. Prevents negative predictions. |
| `cap_multiplier` | float | 1.5 – 5.0 | Cap = max(y) × multiplier. Higher = more headroom for growth. |

### Prophet Core Hyperparameters

| Parameter | Type | Range | Purpose |
|:--|:--|:--|:--|
| `changepoint_prior_scale` | float | 0.001 – 5.0 | Trend flexibility. Higher = faster reaction to trend changes. |
| `seasonality_prior_scale` | float | 0.01 – 50.0 | Seasonality strength. Higher = stronger weekly/monthly patterns. |
| `changepoint_range` | float | 0.70 – 0.95 | Where changepoints are allowed. 0.95 = detects recent changes. |
| `interval_width` | float | 0.70 – 0.95 | Prediction interval width. 0.80 = 80% confidence interval. |
| `n_changepoints` | int | 5 – 80 | Number of potential changepoints. More = finer trend detection. |
| `weekly_seasonality_order` | int | 2 – 10 | Fourier order for weekly patterns. Higher = more flexible shape. |

### Seasonality Parameters

| Parameter | Type | Range | Purpose |
|:--|:--|:--|:--|
| `yearly_seasonality_order` | int | 0 – 10 | Yearly pattern complexity. 0 = disabled. Need > 1 year of data. |
| `monthly_fourier_order` | int | 0 – 5 | Monthly pattern complexity. 0 = disabled. |

### Holiday Parameters

| Parameter | Type | Range | Purpose |
|:--|:--|:--|:--|
| `holiday_prior_scale` | float | 0.1 – 20.0 | Holiday effect strength. Higher = holidays have more impact. |
| `holiday_window_before` | int | 0 – 3 | Days before holiday to include in effect. |
| `holiday_window_after` | int | 0 – 3 | Days after holiday to include in effect. |
