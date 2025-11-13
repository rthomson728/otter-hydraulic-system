# ============================
# Pipeline 3: Feature builder
# ============================
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler

# -------------------------
# Helpers (same semantics)
# -------------------------
def most_frequent(x: pd.Series):
    """Safe mode for resampling booleans/states (mode with NaN-safe fallback)."""
    vc = x.value_counts(dropna=True)
    if vc.empty:
        return np.nan
    return vc.index[0]

def prepare_dataframe_for_resample(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Ensure datetime index for resampling (unchanged logic)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        if time_col not in df.columns:
            raise KeyError(f"'{time_col}' column not found and index is not datetime.")
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=False)
        df = df.set_index(time_col).sort_index()
    return df

def downsample_df(
    df: pd.DataFrame,
    numeric_cols: List[str],
    boolean_cols: List[str],
    rule: str = "15T",
) -> pd.DataFrame:
    """
    Numeric → mean; Boolean → most frequent (then cast to int with NaN→0).
    Mirrors your original 'downsample_15min' behaviour but with arbitrary rule.
    """
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    boolean_cols = [c for c in boolean_cols if c in df.columns]

    parts = []
    if numeric_cols:
        num_ds = df[numeric_cols].resample(rule).mean()
        parts.append(num_ds)
    if boolean_cols:
        bool_ds = df[boolean_cols].resample(rule).agg(most_frequent)
        for c in boolean_cols:
            bool_ds[c] = bool_ds[c].astype("float").round().astype("Int64")
            bool_ds[c] = bool_ds[c].fillna(0).astype(int)
        parts.append(bool_ds)

    if not parts:
        raise ValueError("No matching columns to downsample.")
    return pd.concat(parts, axis=1)

def add_log1p_columns(df: pd.DataFrame, cols: List[str], prefix: str = "log1p_") -> pd.DataFrame:
    """
    Safe log1p that tolerates tiny negatives (clips at -0.999).
    Same semantics as your snippet.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[f"{prefix}{c}"] = np.log1p(out[c].astype(float).clip(lower=-0.999))
    return out

def add_rolling_features(
    df_ds: pd.DataFrame,
    base_numeric_cols: List[str],
    roll_windows: Dict[str, int],
) -> pd.DataFrame:
    """
    Add rolling mean/std for selected windows (in steps), identical to your code.
    std NaNs at start → 0.
    """
    base_numeric_cols = [c for c in base_numeric_cols if c in df_ds.columns]
    out = df_ds.copy()
    for col in base_numeric_cols:
        s = out[col]
        for label, w in roll_windows.items():
            out[f"{col}_mean_{label}"] = s.rolling(window=w, min_periods=1).mean()
            out[f"{col}_std_{label}"]  = s.rolling(window=w, min_periods=1).std()
    std_cols = [c for c in out.columns if any(x in c for x in ["_std_30min", "_std_1h", "_std_2h"])]
    out[std_cols] = out[std_cols].fillna(0)
    return out

def build_feature_matrix(
    df_feat: pd.DataFrame,
    numeric_cols: List[str],
    boolean_cols: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler, List[str]]:
    """
    Assemble X and scale it. Booleans remain as 0/1 and are also scaled.
    Includes all rolling *_mean_*, *_std_* columns discovered in df_feat.
    """
    numeric_cols  = [c for c in numeric_cols  if c in df_feat.columns]
    boolean_cols  = [c for c in boolean_cols  if c in df_feat.columns]
    rolling_cols  = [c for c in df_feat.columns if any(s in c for s in
                     ["_mean_30min","_std_30min","_mean_1h","_std_1h","_mean_2h","_std_2h"])]

    feature_cols  = sorted(list(dict.fromkeys(numeric_cols + boolean_cols + rolling_cols)))
    X = df_feat[feature_cols].copy().fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, scaler, feature_cols

# ---------------------------------------------------------
# Main wrapper: build 15T (or chosen) features end-to-end
# ---------------------------------------------------------
def prepare_features_pipeline3(
    df: pd.DataFrame,
    time_col: str,
    resample_rule: str,
    numeric_features: list,
    boolean_features: list,
    roll_windows: dict,
    log1p_features: Optional[list] = None,
    include_log_roll: bool = True,
    fillna_value: float = 0.0,
):
    """
    End-to-end:
      1) ensure datetime index
      2) resample (numeric=mean, boolean=mode)
      3) add log1p columns (optional)
      4) add rolling means/stds on base + optional log1p columns
      5) build feature matrix + StandardScaler

    Returns:
      df_15         : downsampled dataframe
      df_15_feat    : downsampled + rolled (and log1p if used)
      X             : feature matrix (DataFrame)
      X_scaled      : scaled matrix (np.ndarray)
      scaler        : StandardScaler fit on X
      feature_cols  : list of feature column names used
      meta          : dict of shapes & params for traceability
    """
    # 1) datetime index
    df_dt = prepare_dataframe_for_resample(df, time_col=time_col)

    # 2) resample
    df_15 = downsample_df(
        df_dt,
        numeric_cols=numeric_features,
        boolean_cols=boolean_features,
        rule=resample_rule,
    )

    # 3) log1p (optional, safe for tiny negatives)
    log1p_features = list(log1p_features or [])
    log_bases = [c for c in log1p_features if c in df_15.columns]
    if log_bases:
        df_15 = add_log1p_columns(df_15, log_bases)

    # 4) rolling features
    existing_numeric = [c for c in numeric_features if c in df_15.columns]
    rolling_base = existing_numeric + ([f"log1p_{c}" for c in log_bases] if include_log_roll else [])
    df_15_feat = add_rolling_features(df_15, rolling_base, roll_windows)

    # 5) feature matrix (+ optional booleans if you want them in X)
    numeric_for_model = existing_numeric + [f"log1p_{c}" for c in log_bases]
    # For clustering you often exclude booleans to avoid leakage; pass [] here.
    X, X_scaled, scaler, feature_cols = build_feature_matrix(
        df_15_feat,
        numeric_cols=numeric_for_model,
        boolean_cols=[],  # keep [] unless you explicitly want flags in X
    )

    # Fill any remaining NaNs in df_15/df_15_feat for downstream safety
    df_15 = df_15.fillna(fillna_value)
    df_15_feat = df_15_feat.fillna(fillna_value)

    meta = {
        "params": {
            "time_col": time_col,
            "resample_rule": resample_rule,
            "roll_windows": roll_windows,
            "log1p_features": log_bases,
            "include_log_roll": include_log_roll,
            "fillna_value": fillna_value,
        },
        "shapes": {
            "df_15": tuple(df_15.shape),
            "df_15_feat": tuple(df_15_feat.shape),
            "X": tuple(X.shape),
            "X_scaled": tuple(X_scaled.shape),
        },
        "feature_cols": feature_cols,
    }

    return df_15, df_15_feat, X, X_scaled, scaler, feature_cols, meta