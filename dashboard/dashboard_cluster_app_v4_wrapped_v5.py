# -*- coding: utf-8 -*-
# Streamlit clustering dashboard (HDBSCAN v7, Parquet data with DatetimeIndex)
# Compatible with Python 3.8+
from typing import Optional, List, Tuple, Dict
import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hdbscan import prediction

# ---------------------------------------------------------------------
# Page & Style
# ---------------------------------------------------------------------
st.set_page_config(page_title="Clustering Dashboard (HDBSCAN v7)", layout="wide")
st.markdown("""
<style>
:root, .stApp, .st-emotion-cache-0 { color: #e6e6e6 !important; }
div[data-baseweb="select"] *, ul[role="listbox"] li * { color: #e6e6e6 !important; }
div[data-baseweb="button-group"] button { color: #e6e6e6 !important; }
div[data-baseweb="button-group"] button[aria-pressed="true"] { color: #111 !important; font-weight:600; }
div[role="slider"] * { color: #e6e6e6 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def most_frequent(x: pd.Series):
    if x.empty:
        return np.nan
    vals, counts = np.unique(x.dropna(), return_counts=True)
    if len(vals) == 0:
        return np.nan
    return vals[np.argmax(counts)]

def ensure_dt_index(df: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
    """Ensure DatetimeIndex."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if time_col:
        if time_col not in df.columns:
            raise KeyError(f"'{time_col}' not found. Columns: {list(df.columns)[:12]}…")
        out = df.copy()
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
        out = out.dropna(subset=[time_col]).set_index(time_col).sort_index()
        return out
    raise KeyError("No DatetimeIndex and no valid time column provided.")

def downsample_df(
    df: pd.DataFrame,
    numeric_cols: List[str],
    boolean_cols: List[str],
    rule: str = "15T",
) -> pd.DataFrame:
    num_in  = [c for c in (numeric_cols or []) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    bool_in = [c for c in (boolean_cols or []) if c in df.columns]

    if not num_in and not bool_in:
        num_in = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        bool_in = [c for c in num_in if set(pd.Series(df[c]).dropna().unique()).issubset({0, 1})]
        num_in = [c for c in num_in if c not in bool_in]

    parts = []
    if num_in:
        parts.append(df[num_in].resample(rule).mean())
    if bool_in:
        bf = df[bool_in].resample(rule).agg(most_frequent)
        for c in bool_in:
            bf[c] = pd.to_numeric(bf[c], errors="coerce").fillna(0).round().astype(int)
        parts.append(bf)

    if not parts:
        raise ValueError("No matching columns to downsample.")
    out = pd.concat(parts, axis=1)
    out = out.loc[:, ~out.columns.duplicated(keep="first")]
    return out

def add_log1p(df_15: pd.DataFrame, bases: List[str]) -> pd.DataFrame:
    out = df_15.copy()
    present = [c for c in bases if c in out.columns]
    for c in present:
        # FIX: use lower= (not min=)
        out[f"log1p_{c}"] = np.log1p(
            out[c].astype(float).clip(lower=-0.999)
        )
    return out

def add_rolls(df_15: pd.DataFrame, bases: List[str], roll_windows: Dict[str, int]) -> pd.DataFrame:
    out = df_15.copy()
    bases = [c for c in bases if c in out.columns]
    for label, win in (roll_windows or {}).items():
        if win and win > 0 and bases:
            m = out[bases].rolling(window=win, min_periods=win)
            out[[f"{c}_mean_{label}" for c in bases]] = m.mean().values
            out[[f"{c}_std_{label}"  for c in bases]] = m.std(ddof=0).values
    return out

def align_for_transform(X_df: pd.DataFrame, scaler: StandardScaler, meta: dict) -> pd.DataFrame:
    if hasattr(scaler, "feature_names_in_") and len(scaler.feature_names_in_) > 0:
        expected = list(scaler.feature_names_in_)
    else:
        expected = list(meta.get("feature_cols", [])) or list(meta.get("feature_names_in_", []))
    if not expected:
        raise ValueError("No expected feature columns found in scaler/meta.")

    X = X_df.copy()
    for c in [c for c in expected if c not in X.columns]:
        X[c] = 0.0
    X = X.reindex(columns=expected)
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.astype(np.float64)

def build_features_from_meta(
    df_raw: pd.DataFrame,
    meta: dict,
    time_col: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuilds 15min features per meta."""
    df_raw = ensure_dt_index(df_raw, time_col=time_col)
    rule = meta.get("resample_rule", "15T")
    num_req  = list(meta.get("numeric_features_requested", []))
    bool_req = list(meta.get("boolean_features_requested", []))
    log_req  = list(meta.get("log1p_features_requested", []))
    roll_windows = meta.get("roll_windows", {"30min": 2, "1h": 4, "2h": 8})

    num_in  = [c for c in num_req  if c in df_raw.columns]
    bool_in = [c for c in bool_req if c in df_raw.columns]

    if not num_in and not bool_in:
        raise ValueError("Meta features missing from data.")

    df_15 = downsample_df(df_raw, num_in, bool_in, rule=rule)
    log_bases = [c for c in log_req if c in df_15.columns]
    df_15_log = add_log1p(df_15, log_bases)
    roll_bases = num_in + [f"log1p_{c}" for c in log_bases]
    df_15_feat = add_rolls(df_15_log, roll_bases, roll_windows)
    df_15_feat = df_15_feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df_15, df_15_feat

def run_inference_hdbscan(
    df_raw: pd.DataFrame,
    model,
    xform: dict,
    meta: dict,
    cluster_map: Dict[int, str],
    start=None,
    end=None,
) -> pd.DataFrame:
    """Apply HDBSCAN model (approximate_predict)."""
    if start or end:
        if isinstance(df_raw.index, pd.DatetimeIndex):
            df_raw = df_raw.loc[start:end]

    df_15, df_15_feat = build_features_from_meta(df_raw, meta, time_col=meta.get("time_col", None))
    scaler: StandardScaler = xform["scaler"]
    pca: PCA = xform["pca"]

    X_aligned = align_for_transform(df_15_feat, scaler, meta)
    if X_aligned.shape[0] == 0:
        return pd.DataFrame(index=df_15_feat.index, columns=["cluster", "cluster_probability", "state"])

    X_scaled = scaler.transform(X_aligned)
    X_pca = pca.transform(X_scaled)
    labels, probs = prediction.approximate_predict(model, X_pca)
    out = pd.DataFrame({"cluster": labels, "cluster_probability": probs}, index=df_15_feat.index)
    out["state"] = out["cluster"].map(lambda cid: cluster_map.get(int(cid), "Noise" if int(cid) == -1 else "Unknown"))
    return out

# ---------------------------------------------------------------------
# Cached Loaders
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts(model_path: str, xform_path: str, meta_path: str, cluster_map_path: str):
    import joblib
    model = joblib.load(model_path)
    xform = joblib.load(xform_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    with open(cluster_map_path, "r") as f:
        cluster_map = json.load(f)
    cluster_map = {int(k): v for k, v in cluster_map.items()}
    return model, xform, meta, cluster_map

@st.cache_data
def load_parquet_or_csv(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python")
    return df

# ---------------------------------------------------------------------
# Sidebar Config
# ---------------------------------------------------------------------
st.sidebar.header("Artifacts")
default_root = Path(r"C:\Users\rosst\OneDrive\Control Integrity\otter-hydraulic-system")

model_path = st.sidebar.text_input("HDBSCAN model (.pkl)", str(default_root / "notebooks" / "models" / "hdbscan_pca_model_v7.pkl"))
xform_path = st.sidebar.text_input("Scaler+PCA (.pkl)", str(default_root / "notebooks" / "models" / "preprocess_scaler_pca_v7.pkl"))
meta_path = st.sidebar.text_input("Meta (.json)", str(default_root / "notebooks" / "models" / "hdbscan_meta_v7.json"))
cmap_path = st.sidebar.text_input("Cluster map (.json)", str(default_root / "notebooks" / "models" / "cluster_map_v7.json"))

st.sidebar.header("Data")
data_path = st.sidebar.text_input(
    "Parquet/CSV path",
    r"C:\Users\rosst\OneDrive\Control Integrity\otter-hydraulic-system\data\df_all_otter_labeled_cleaned_v1_250825.parquet"
)
time_col_sidebar = st.sidebar.text_input("Timestamp column (leave blank to use existing index)", "")

st.sidebar.header("Window")
quick = st.sidebar.selectbox("Quick window", ["All", "1 day", "3 days", "7 days", "14 days", "30 days"], index=2)
start_dt = st.sidebar.text_input("Start (YYYY-MM-DD HH:MM)", "")
end_dt   = st.sidebar.text_input("End   (YYYY-MM-DD HH:MM)", "")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    st.title("Clustering Dashboard — HDBSCAN (v7 meta-aligned)")

    # Load artifacts
    try:
        model, xform, meta, cluster_map = load_artifacts(model_path, xform_path, meta_path, cmap_path)
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        return

    # Load data
    try:
        df_raw = load_parquet_or_csv(data_path)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    # Ensure datetime index
    time_col = time_col_sidebar.strip() if time_col_sidebar else None
    try:
        df = ensure_dt_index(df_raw, time_col=time_col)
    except Exception as e:
        st.error(f"Failed to set DatetimeIndex: {e}")
        st.dataframe(df_raw.head(20))
        return

    # Determine window
    start, end = None, None
    if quick != "All":
        if df.index.size == 0:
            st.warning("No data in file.")
            return
        end = df.index.max()
        days = int(quick.split()[0])
        start = end - pd.Timedelta(days=days)
    if start_dt.strip():
        start = pd.to_datetime(start_dt, errors="coerce")
    if end_dt.strip():
        end = pd.to_datetime(end_dt, errors="coerce")

    st.write(f"Data window: {start or df.index.min()} → {end or df.index.max()}  (rows: {len(df.loc[start:end])})")

    # Run inference
    try:
        meta = dict(meta)
        meta["time_col"] = None  # use DatetimeIndex
        pred = run_inference_hdbscan(df, model, xform, meta, cluster_map, start=start, end=end)
    except Exception as e:
        st.error(f"Inference failed: {e}")
        return

    if pred.empty or pred["state"].isna().all():
        st.warning("No predictions available. Try a longer time range.")
        st.dataframe(pred)
        return

    # Current state
    last_row = pred.dropna().iloc[-1]
    current_state = str(last_row.get("state", "Unknown"))
    current_prob  = float(last_row.get("cluster_probability", np.nan))
    current_cluster = int(last_row.get("cluster", -1))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Current State", current_state)
    with c2:
        st.metric("Cluster", f"{current_cluster}")
    with c3:
        st.metric("Cluster Prob.", f"{current_prob:.2f}" if not np.isnan(current_prob) else "—")

    # Plot
    st.subheader("State Timeline (15-min)")
    st.line_chart(pred[["cluster_probability"]])
    st.dataframe(pred.tail(200))

    # Meta info
    with st.expander("Debug / Meta"):
        st.json({
            "resample_rule": meta.get("resample_rule", "15T"),
            "roll_windows": meta.get("roll_windows"),
            "numeric_requested_sample": meta.get("numeric_features_requested", [])[:10],
            "boolean_requested_sample": meta.get("boolean_features_requested", [])[:10],
            "log1p_requested_sample": meta.get("log1p_features_requested", [])[:10],
        })

if __name__ == "__main__":
    main()
