# dashboard_app_v20_hdbscan.py
# v20 dashboard (same functionality & look) but using your HDBSCAN clustering model
# with auto cluster→state detection when a mapping isn't present.
import sys
from pathlib import Path

PIPELINE_DIR = Path(r"C:\Users\rosst\OneDrive\Control Integrity\otter-hydraulic-system\pipeline")
if PIPELINE_DIR.exists():
    p = str(PIPELINE_DIR.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
else:
    raise FileNotFoundError(f"Pipeline directory not found: {PIPELINE_DIR}")
from typing import Optional, Tuple, Dict
import streamlit as st

def _safe_set_page_config(*args, **kwargs):
    try:
        st.set_page_config(*args, **kwargs)
    except Exception:
        pass

def main():
    # ----------------------------
    # Imports (local scope)
    # ----------------------------
    from pathlib import Path
    import json, joblib, re
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.io as pio
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from hdbscan import prediction as hdb_pred

    # Optional extras
    try:
        from streamlit_extras.app_autorefresh import st_autorefresh
        HAVE_REFRESH = True
    except Exception:
        HAVE_REFRESH = False

    try:
        from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
        HAVE_AGGRID = True
    except Exception:
        HAVE_AGGRID = False

    # === Your pipeline imports (unchanged) ===
    from otter_pipeline_function_definitions import (
        run_full_pipeline,
        clean_dataframe_and_split_valves,
    )
    from otter_pipeline__labelling_function_definitions import run_labelling_pipeline
    import otter_config as cfg

    # ----------------------------
    # Paths to HDBSCAN artifacts
    # ----------------------------
    MODEL_DIR_DEFAULT = Path(r"C:\Users\rosst\OneDrive\Control Integrity\otter-hydraulic-system\notebooks\models")
    HDBSCAN_MODEL_FN  = "hdbscan_pca_model_v7.pkl"
    XFORM_FN          = "preprocess_scaler_pca_v7.pkl"   # {"scaler":..., "pca":...}
    META_FN           = "hdbscan_meta_v7.json"           # has feature_names, boolean_features, roll_windows, resample_rule, (optional) cluster_to_state

    # ----------------------------
    # Data path (unchanged)
    # ----------------------------
    PARQUET_PATH = r"C:\Users\rosst\OneDrive\Control Integrity\Data\Otter 2003 to 2024 with tanks B\Parquet_Clean_Output\Otter_All_Combined_PCV_180825.parquet"

    # ----------------------------
    # v20 UI constants (unchanged)
    # ----------------------------
    COLUMNS_NEEDED = [
        "HPU_SPLY_LEV_L","HPU_RET_LEV_L",
        "HPU_LPA_OUT","HPU_LPB_OUT","HPU_HPA_OUT","HPU_HPB_OUT",
        "SCM1_LP_CONS","SCM1_HP_CONS",
        "Supply_Consumption_Excl_Fills","External_Losses",
        "FCV_Fluid_Usage","Valve_Operation_Fluid","umbilical_charge_volume",
        "LP_Runs_24h","LP_Runtime_24h_min","HP_Runtime_24h_min",
        "LP_Runtime_Cumulative_min","HP_Runtime_Cumulative_min",
        "LP_pump_state","HP_pump_state","HP_pump_rate_2h",
        "baseline_drop_L",
        "is_steady_state","is_lp_low","is_hp_low","is_lp_high","is_hp_high",
        "is_no_redundancy","is_supply_tank_low","is_pressurising",
        "is_fcv_ops","is_losses_high",
        "Valve_Event_Log",
    ]

    # Good looking dark theme (same as v20)
    _safe_set_page_config(page_title="Otter Hydraulic Dashboard", layout="wide", page_icon="💧")
    PAPER_BG = "#0e1117"; PLOT_BG = "#0e1117"; GRID_COL = "#2a2f3a"; FONT_COL = "#e8f0f6"
    pio.templates["otter_darkgrey"] = pio.templates["plotly_dark"]
    tpl = pio.templates["otter_darkgrey"].layout
    tpl.colorway = ["#1f77b4","#2ca02c","#ff7f0e","#d62728","#9467bd","#17becf"]
    tpl.paper_bgcolor = PAPER_BG; tpl.plot_bgcolor = PLOT_BG; tpl.font.color = FONT_COL
    pio.templates.default = "otter_darkgrey"

    # Buttons + CSS (same as v20)
    st.markdown("""
    <style>
    .main, .stApp { background-color:#000 !important; }
    [data-testid="stHeader"] { background-color:#000 !important; }
    h1#otter-hydraulic-dashboard { font-size:1.28rem !important; margin:0.6rem 0 0.35rem 0 !important; color:#e8f0f6 !important; }
    h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown { color:#e8f0f6 !important; }
    .block-container { padding: 0.7rem 0.8rem 0.6rem 0.8rem !important; }
    section[data-testid="stSidebar"] { background:#0e0e0e !important; width:240px !important; min-width:240px !important; }
    section[data-testid="stSidebar"] * { color:#d5d5d5 !important; }
    div[data-testid="stHorizontalBlock"] { gap:0.35rem !important; }
    .kpi-card { background:#0b0b0b; border:1px solid #303030; border-radius:10px; padding:8px 10px; text-align:center; min-height:56px; }
    .kpi-title { color:#a8b4bf; font-size:0.72rem; margin-bottom:2px; }
    .kpi-value { color:#e8f0f6; font-size:1.05rem; font-weight:800; line-height:1.0; }
    .kpi-sub { margin-top:4px; color:#cfe0ea; font-size:0.83rem; font-weight:650; }
    .kpi-ok{ background:#0f241a; border-color:#1f7a4a; } .kpi-warn{ background:#382900; border-color:#ffbf00; } .kpi-bad{ background:#2a0c14; border-color:#b02236; } .kpi-neu{background:#0b0b0b; border-color:#303030;}
    .square{ width:14px; height:14px; border-radius:3px; border:1px solid #303030; display:inline-block; margin-right:6px; }
    .square-green { background:#22c55e; } .square-red{ background:#ef4444; }
    .status-panel{ border:1px solid #303030; border-radius:10px; background:#0b0b0b; padding:10px; }
    .badge{ display:inline-flex; align-items:center; padding:4px 8px; margin:4px 6px 4px 0; border-radius:8px; font-size:0.78rem; font-weight:600; background:#111; border:1px solid #303030; color:#bfd4e4; }
    .dot{ width:8px; height:8px; border-radius:50%; margin-right:6px; } .dot-ok{background:#25d366;} .dot-bad{background:#ff4d4f;} .dot-warn{background:#f0a500;}
    button[kind="primary"], button[kind="secondary"], .stButton > button, form.stForm button {
        background-color:#000 !important; color:#fff !important; border:1px solid #444 !important; border-radius:6px !important; font-weight:600 !important; padding:0.5em 1.2em !important;
    }
    button[kind="primary"]:hover, button[kind="secondary"]:hover, .stButton > button:hover, form.stForm button:hover {
        background-color:#1a1a1a !important; border:1px solid #666 !important; color:#fff !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ----------------------------
    # Sidebar controls (same knobs)
    # ----------------------------
    st.sidebar.title("Controls")
    preset = st.sidebar.selectbox("Quick window", ["12 hours","1 day","7 days","30 days","90 days","1 year"], index=2)
    resample_rule = st.sidebar.selectbox("Chart resolution", ["1min","2min","5min","10min","15min"], index=2)
    use_custom = st.sidebar.checkbox("Use custom date range", value=False)

    if use_custom:
        c1, c2 = st.sidebar.columns(2)
        start_date = c1.date_input("Start date")
        end_date   = c2.date_input("End date")
    else:
        start_date = end_date = None

    refresh_ms = st.sidebar.slider("Auto-refresh (ms)", 0, 300000, 120000, 5000)

    # ----------------------------
    # Helpers (same as v20)
    # ----------------------------
    def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        return df[~df.index.isna()].sort_index()

    def _window_range_from_preset(df: pd.DataFrame, label: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
        end = df.index.max()
        if label == "12 hours": start = end - pd.Timedelta(hours=12)
        elif label == "1 day":  start = end - pd.Timedelta(days=1)
        elif label == "7 days": start = end - pd.Timedelta(days=7)
        elif label == "30 days": start = end - pd.Timedelta(days=30)
        elif label == "90 days": start = end - pd.Timedelta(days=90)
        elif label == "1 year":  start = end - pd.DateOffset(years=1)
        else: start = end - pd.Timedelta(days=7)
        return start, end

    def downsample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        df = _ensure_dt_index(df.copy())
        if df.empty: return df
        num = df.select_dtypes(include=["number"])
        boo = df.select_dtypes(include=["bool"]).astype(int)
        agg_num = num.resample(rule).mean() if not num.empty else pd.DataFrame(index=df.index)
        agg_boo = boo.resample(rule).max().astype(bool) if not boo.empty else pd.DataFrame(index=df.index)
        out = pd.concat([agg_num, agg_boo], axis=1)
        if "Valve_Event_Log" in df.columns:
            txt = df[["Valve_Event_Log"]].copy()
            txt["Valve_Event_Log"] = txt["Valve_Event_Log"].fillna("")
            out = out.join(txt.resample(rule).last(), how="left")
        return out

    def kpi_card(title: str, value: str, tone: str="neu", subtitle: Optional[str] = None):
        tone_cls = {"ok":"kpi-ok","warn":"kpi-warn","bad":"kpi-bad","neu":"kpi-neu"}.get(tone,"kpi-neu")
        sub_html = f'<div class="kpi-sub">{subtitle}</div>' if subtitle else ""
        st.markdown(f"""
        <div class="kpi-card {tone_cls}">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          {sub_html}
        </div>
        """, unsafe_allow_html=True)

    # ----------------------------
    # HDBSCAN loader + feature builder (for your model)
    # ----------------------------
    @st.cache_resource(show_spinner=False)
    def load_hdbscan_suite(model_dir: Optional[str] = None):
        """
        Load HDBSCAN model, scaler+pca, and meta.
        """
        cand_dir = Path(model_dir) if model_dir else MODEL_DIR_DEFAULT
        model_path = cand_dir / HDBSCAN_MODEL_FN
        xform_path = cand_dir / XFORM_FN
        meta_path  = cand_dir / META_FN
        if not (model_path.exists() and xform_path.exists() and meta_path.exists()):
            st.error(f"Model artifacts not found in: {cand_dir}\nExpected: {HDBSCAN_MODEL_FN}, {XFORM_FN}, {META_FN}")
            return None, None, None
        try:
            model = joblib.load(model_path)
            xform = joblib.load(xform_path)       # {"scaler":..., "pca":...}
            meta  = json.loads(meta_path.read_text())
            return model, xform, meta
        except Exception as e:
            st.error(f"Failed loading HDBSCAN suite: {e}")
            return None, None, None

    # Feature helpers (match your training meta)
    def most_frequent(x: pd.Series):
        vc = x.value_counts(dropna=True)
        return vc.index[0] if not vc.empty else np.nan

    def _is_numeric_series(s: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(s)

    def downsample_15min(df: pd.DataFrame, numeric_cols: list, boolean_cols: list, rule: str = "15T"):
        numeric_cols = [c for c in numeric_cols if c in df.columns and _is_numeric_series(df[c])]
        boolean_cols = [c for c in boolean_cols if c in df.columns]
        parts = []
        if numeric_cols:
            parts.append(df[numeric_cols].resample(rule).mean())
        if boolean_cols:
            bool_ds = df[boolean_cols].resample(rule).agg(most_frequent)
            for c in boolean_cols:
                bool_ds[c] = bool_ds[c].astype("float").round().astype("Int64").fillna(0).astype(int)
            parts.append(bool_ds)
        if not parts: return pd.DataFrame(index=df.index)
        out = pd.concat(parts, axis=1)
        return out.loc[:, ~out.columns.duplicated(keep="first")]

    def add_log1p_columns(df: pd.DataFrame, cols: list, prefix: str = "log1p_"):
        df = df.copy()
        for c in cols:
            if c in df.columns:
                df[f"{prefix}{c}"] = np.log1p(df[c].astype(float).clip(lower=-0.999))
        return df

    def add_rolling_features(df_ds: pd.DataFrame, base_numeric_cols: list, roll_windows: dict):
        cols = [c for c in base_numeric_cols if c in df_ds.columns]
        if not cols: return df_ds.copy()
        out_parts = [df_ds]
        for label, w in roll_windows.items():
            roll = df_ds[cols].rolling(window=w, min_periods=1)
            mean_df = roll.mean()
            std_df  = roll.std().fillna(0)
            mean_df.columns = [f"{c}_mean_{label}" for c in cols]
            std_df.columns  = [f"{c}_std_{label}"  for c in cols]
            out_parts.extend([mean_df, std_df])
        return pd.concat(out_parts, axis=1)

    def build_feature_matrix(df_feat: pd.DataFrame, numeric_cols: list, boolean_cols: list, rolling_suffixes=(" _mean_30min","_std_30min","_mean_1h","_std_1h","_mean_2h","_std_2h")):
        numeric_cols = [c for c in numeric_cols if c in df_feat.columns]
        boolean_cols = [c for c in boolean_cols if c in df_feat.columns]
        rolling_cols = [c for c in df_feat.columns if any(s in c for s in ("_mean_30min","_std_30min","_mean_1h","_std_1h","_mean_2h","_std_2h"))]
        feat_cols = sorted(list(dict.fromkeys(numeric_cols + boolean_cols + rolling_cols)))
        X = df_feat[feat_cols].copy().fillna(0)
        return X, feat_cols

    def extract_base_feature_lists(meta: dict, df_raw: pd.DataFrame):
        feat_order = meta.get("feature_names", [])
        bool_feats = set(meta.get("boolean_features", []))
        base_numeric = [
            c for c in feat_order
            if (c in df_raw.columns) and (c not in bool_feats)
            and (not c.startswith("log1p_"))
            and (not c.endswith(("_mean_30min","_std_30min","_mean_1h","_std_1h","_mean_2h","_std_2h")))
            and _is_numeric_series(df_raw[c])
        ]
        base_boolean = [c for c in bool_feats if c in df_raw.columns]
        return list(dict.fromkeys(base_numeric)), list(dict.fromkeys(base_boolean))

    # ----------------------------
    # AUTO label rules (NOT hard-coded to ids)
    # ----------------------------
    AUTO_PARAMS: Dict[str, float] = {
        "steady_state_pct_strict" : 100.0,  # System Depressurised proxy
        "steady_state_pct_hi"     : 90.0,   # steady dominance
        "fcv_ops_pct_min"         : 75.0,   # FCV dominance
        "losses_pct_min"          : 40.0,   # losses high
        "hp_pump_rate_min"        : 1.10,
        "lp_pump_rate_min"        : 1.00,
        "min_nonnoise_points"     : 300,
        "probability_floor"       : 0.70,
        "steady_downweight_loss"  : 5.0,
    }
    AUTO_PARAMS_NOISE: Dict[str, object] = {
        "purity_min": 0.70,
        "purity_keys": ["is_steady_state_pct","is_fcv_ops_pct","is_losses_high_pct"],
        "mixed_steady_low": 40.0, "mixed_steady_high": 70.0,
        "mixed_fcv_min": 8.0, "mixed_losses_max": 30.0,
        "mean_prob_min": 0.70, "min_points": 500,
    }

    def _cluster_summary(df_pred: pd.DataFrame) -> pd.DataFrame:
        cols = set(df_pred.columns)
        need_flags = ["is_fcv_ops","is_tank_fill","is_steady_state","is_steady_state_strict","is_losses_high"]
        have_flags = [c for c in need_flags if c in cols]
        g = df_pred.groupby("cluster").agg(
            n_points=("cluster","size"),
            mean_prob=("cluster_probability","mean"),
            HP_pump_rate_2h_mean=("HP_pump_rate_2h","mean") if "HP_pump_rate_2h" in cols else ("cluster","size"),
            LP_pump_rate_2h_mean=("LP_pump_rate_2h","mean") if "LP_pump_rate_2h" in cols else ("cluster","size"),
        ).fillna(0.0)
        for c in have_flags:
            g[f"{c}_pct"] = 100.0 * df_pred.groupby("cluster")[c].mean()
        g.insert(0,"cluster_id", g.index.astype(int))
        for k in ["is_fcv_ops_pct","is_tank_fill_pct","is_steady_state_pct","is_steady_state_strict_pct","is_losses_high_pct","HP_pump_rate_2h_mean","LP_pump_rate_2h_mean"]:
            if k not in g.columns: g[k] = 0.0
        return g.reset_index(drop=True)

    def _is_noise_like_row(r: pd.Series, p=AUTO_PARAMS_NOISE) -> bool:
        if int(r["cluster_id"]) == -1: return True
        purity_vals = [float(r.get(k,0.0))/100.0 for k in p["purity_keys"] if k in r.index]
        low_purity = (max(purity_vals) if purity_vals else 0.0) < float(p["purity_min"])
        steady, fcv, losses = float(r.get("is_steady_state_pct",0.0)), float(r.get("is_fcv_ops_pct",0.0)), float(r.get("is_losses_high_pct",0.0))
        mixed = (float(p["mixed_steady_low"]) <= steady <= float(p["mixed_steady_high"])) and (fcv >= float(p["mixed_fcv_min"])) and (losses <= float(p["mixed_losses_max"]))
        low_prob  = float(r.get("mean_prob",0.0)) < float(p["mean_prob_min"])
        too_small = int(r.get("n_points",0)) < int(p["min_points"])
        return (low_purity and (low_prob or too_small)) or (mixed and (low_prob or too_small))

    def _score_and_label(summary: pd.DataFrame, P=AUTO_PARAMS) -> pd.DataFrame:
        s = summary.copy()

        def sys_dep(r):
            base = 1.0 if r["is_steady_state_pct"] >= P["steady_state_pct_strict"]-1e-6 else 0.0
            pumps_off = float(r["HP_pump_rate_2h_mean"] < 0.2 and r["LP_pump_rate_2h_mean"] < 0.2)
            return base + 0.2*pumps_off

        def fcv(r):
            base = r["is_fcv_ops_pct"]/100.0
            base += 0.1*(r["mean_prob"] >= P["probability_floor"])
            base += 0.1*(r["n_points"] >= P["min_nonnoise_points"])
            if r["is_fcv_ops_pct"] >= P["fcv_ops_pct_min"]:
                base *= 1.2
            return base

        def steady(r):
            s_ = (r["is_steady_state_pct"]/100.0)
            if r["is_losses_high_pct"] >= P["losses_pct_min"]:
                s_ = s_ / P["steady_downweight_loss"]
            s_ += 0.05*(r["mean_prob"] >= P["probability_floor"])
            if r["is_steady_state_pct"] >= P["steady_state_pct_hi"]:
                s_ *= 1.2
            return s_

        def hp_loss(r):
            return 1.0 if (r["is_losses_high_pct"]>=P["losses_pct_min"] and r["HP_pump_rate_2h_mean"]>=P["hp_pump_rate_min"]) else 0.0

        def lp_loss(r):
            return 1.0 if (r["is_losses_high_pct"]>=P["losses_pct_min"] and r["LP_pump_rate_2h_mean"]>=P["lp_pump_rate_min"]) else 0.0

        s["score_sys_depr"] = s.apply(sys_dep, axis=1)
        s["score_fcv"]      = s.apply(fcv, axis=1)
        s["score_steady"]   = s.apply(steady, axis=1)
        s["score_hp_loss"]  = s.apply(hp_loss, axis=1)
        s["score_lp_loss"]  = s.apply(lp_loss, axis=1)

        labels, confs = [], []
        for _, r in s.iterrows():
            cid = int(r["cluster_id"])
            if cid == -1 or _is_noise_like_row(r):
                labels.append("Noise"); confs.append(1.0 if cid == -1 else 0.8); continue
            cand = {
                "System Depressurised": r["score_sys_depr"],
                "FCV Operation"       : r["score_fcv"],
                "HP Losses"           : r["score_hp_loss"],
                "LP Losses"           : r["score_lp_loss"],
                "Steady State"        : r["score_steady"],
            }
            best_label = max(cand, key=cand.get); best_score = float(cand[best_label])
            if best_score <= 0.0:
                best_label = "Steady State" if r["is_steady_state_pct"] >= P["steady_state_pct_hi"] else "Other"
                best_score = 0.3 if best_label=="Steady State" else 0.1
            conf = float(np.clip(best_score, 0, 1))
            conf += 0.05*(r["mean_prob"] >= P["probability_floor"])
            conf += 0.05*(r["n_points"] >= P["min_nonnoise_points"])
            labels.append(best_label); confs.append(round(float(np.clip(conf,0,1)),3))
        s["auto_state"] = labels; s["auto_conf"] = confs
        return s

    def auto_mapping_from_pred(pred: pd.DataFrame) -> Tuple[Dict[int,str], pd.DataFrame]:
        summary = _cluster_summary(pred)
        scored  = _score_and_label(summary)
        mapping = {int(r.cluster_id): r.auto_state for r in scored.itertuples()}
        return mapping, scored

    # ----------------------------
    # HDBSCAN inference on a time window (15-min bins)
    # ----------------------------
    @st.cache_data(show_spinner=False)
    def cluster_states_15min(
        df_raw: pd.DataFrame,
        _model, _xform, _meta,
        start: pd.Timestamp, end: pd.Timestamp,
    ):
        # restrict
        df = _ensure_dt_index(df_raw.loc[start:end].copy())
        if df.empty:
            return pd.DataFrame(index=pd.DatetimeIndex([]), columns=["state","cluster","cluster_probability"])

        # Base lists from meta
        base_numeric_raw, base_boolean_raw = extract_base_feature_lists(_meta, df)

        # Downsample to model’s resolution
        rule = _meta.get("resample_rule", "15T")
        df_15 = downsample_15min(df, base_numeric_raw, base_boolean_raw, rule)

        # log1p
        log_bases = [c for c in _meta.get("log1p_features", []) if c in df_15.columns]
        df_15 = add_log1p_columns(df_15, log_bases)

        # rolling
        base_numeric_for_roll = [c for c in base_numeric_raw if c in df_15.columns]
        base_numeric_for_roll += [f"log1p_{c}" for c in log_bases if f"log1p_{c}" in df_15.columns]
        df_15_feat = add_rolling_features(df_15, base_numeric_for_roll, _meta.get("roll_windows", {"30min":2,"1h":4,"2h":8}))

        # Exact feature order
        feat_order = _meta.get("feature_names", [])
        bool_feats = _meta.get("boolean_features", [])
        X_all, _ = build_feature_matrix(df_15_feat, numeric_cols=feat_order, boolean_cols=bool_feats)
        X_df = X_all.reindex(columns=feat_order).fillna(_meta.get("fillna_value", 0))

        # Transform: scaler → PCA
        scaler: StandardScaler = _xform["scaler"]; pca: PCA = _xform["pca"]

        # Determine expected column order (prefer what the scaler was fit on)
        if hasattr(scaler, "feature_names_in_") and len(getattr(scaler, "feature_names_in_", [])) > 0:
            expected_cols = list(scaler.feature_names_in_)
        else:
            expected_cols = list(feat_order)  # from meta; must be non-empty

        # Guard: if we somehow have no expected columns, return empty result
        if not expected_cols:
            return pd.DataFrame(index=df_15_feat.index, columns=["state","cluster","cluster_probability"])

        # Align X_df to expected schema: add missing as 0.0, drop extras, order cols
        missing = [c for c in expected_cols if c not in X_df.columns]
        if missing:
            for c in missing:
                X_df[c] = 0.0
        X_df = X_df.reindex(columns=expected_cols)

        # Coerce to numeric float, replace inf/nan
        X_df = X_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)

        # If no rows after windowing/feature build, short-circuit
        if X_df.shape[0] == 0:
            return pd.DataFrame(index=df_15_feat.index, columns=["state","cluster","cluster_probability"])

        # Now safe to transform
        X_scaled = scaler.transform(X_df)
        X_pca = pca.transform(X_scaled)

        # Predict clusters
        labels_raw, strengths = hdb_pred.approximate_predict(_model, X_pca)
        remap = _meta.get("label_remap", {})
        labels = np.array([remap.get(int(x), int(x)) for x in labels_raw], dtype=int)

        out = df_15_feat.copy()
        out["cluster"] = labels
        out["cluster_probability"] = strengths

        # Map to states — prefer saved mapping; else auto
        c2s = None
        if hasattr(_model, "cluster_to_state_"):
            c2s = {int(k): str(v) for k, v in getattr(_model, "cluster_to_state_").items()}
        elif _meta.get("cluster_to_state"):
            c2s = {int(k): str(v) for k, v in _meta["cluster_to_state"].items()}

        scored = None
        if not c2s:
            c2s, scored = auto_mapping_from_pred(out)

        out["state"] = out["cluster"].map(c2s).fillna("Other")
        return out[["state","cluster","cluster_probability"]]

    # ----------------------------
    # Load & run your existing pipeline (unchanged)
    # ----------------------------
    @st.cache_data(ttl=120, show_spinner=True)
    def load_and_process(parquet_path: str, years_back: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_all = run_full_pipeline(
            parquet_path=parquet_path,
            columns_to_remove=cfg.columns_to_remove,
            valve_columns=cfg.valve_columns,
            valve5_volume=cfg.valve5_volume,
            valve2_volume=cfg.valve2_volume,
            valve_hp_volume=cfg.valve_hp_volume,
            pcv_columns=cfg.pcv_columns,
            thresholds=cfg.thresholds,
            pump_events_path=cfg.pump_events_path,
            valve_transition_cols=cfg.valve_transition_cols,
            drop_slope_data_path=cfg.drop_slope_data_path,
            use_smoothed_tank=True, smoothing_strength=5, combine_sensors="both",
            years_back=years_back,
        )
        labeled_df, _ = run_labelling_pipeline(
            df_all,
            channel_thresholds=cfg.UMBILICAL_THRESHOLDS,
            pressurising_level_col='HPU_SPLY_LEV_L',
            plot=False,
        )
        df_clean_all, _ = clean_dataframe_and_split_valves(labeled_df)
        df_clean_all = _ensure_dt_index(df_clean_all)
        df_full = df_clean_all.copy()
        have = [c for c in COLUMNS_NEEDED if c in df_clean_all.columns]
        df_clean = df_clean_all[have]
        return df_clean, df_full

    if HAVE_REFRESH and refresh_ms and refresh_ms > 0:
        st_autorefresh(interval=refresh_ms, key="auto_refresh")

    with st.spinner("Running pipeline…"):
        df_all, df_full = load_and_process(PARQUET_PATH, years_back=2)

    if df_all.empty:
        st.error("No data returned from pipeline."); st.stop()

    # Window selection (unchanged)
    if use_custom and start_date and end_date:
        start = pd.Timestamp(start_date)
        end   = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        start, end = _window_range_from_preset(df_all, preset)

    df_view = _ensure_dt_index(df_all.loc[start:end].copy())
    if df_view.empty:
        st.markdown('<h1 id="otter-hydraulic-dashboard">Otter Hydraulic Dashboard</h1>', unsafe_allow_html=True)
        st.warning("No data in the selected window."); st.stop()

    PAD_DAYS = 400
    calc_start = max(df_all.index.min(), (df_view.index.min() - pd.Timedelta(days=PAD_DAYS)))
    calc_end   = df_all.index.max()
    df_calc = df_all.loc[calc_start:calc_end].copy()

    # Title
    st.markdown('<h1 id="otter-hydraulic-dashboard">Otter Hydraulic Dashboard</h1>', unsafe_allow_html=True)
    _now = pd.Timestamp.now().tz_localize(None)
    st.caption(f"Current Date/Time: {_now:%d-%m-%Y %H:%M}")
    range_txt_sel = f"{df_view.index.min():%d-%m-%Y} → {df_view.index.max():%d-%m-%Y}"

    # ----------------------------
    # KPIs row (same visuals), current state via HDBSCAN
    # ----------------------------
    k0, k1, k2, k3, k4, k5, k6, k7 = st.columns(8)

    # Load HDBSCAN artifacts once
    model, xform, meta = load_hdbscan_suite(str(MODEL_DIR_DEFAULT))

    with k0:
        current_label = "—"; tone = "neu"
        if model is not None and xform is not None and meta is not None:
            end_win = df_view.index.max()
            start_win = end_win - pd.Timedelta(hours=2)           # last 2h → get the last 15-min bin
            pred_15 = cluster_states_15min(df_full, model, xform, meta, start=start_win, end=end_win)
            if not pred_15.empty:
                current_label = str(pred_15["state"].iloc[-1])
                # tone mapping
                if current_label == "Steady State": tone = "ok"
                elif current_label in ("System Depressurised","FCV Operation"): tone = "warn"
                elif current_label in ("HP Losses","LP Losses","Noise","Other"): tone = "bad"
                else: tone = "neu"
        kpi_card("Current State", current_label, tone=tone, subtitle="last 15-min bin")

    # Flow + daily totals (same as v20)
    end_window = df_view.index.max()
    start_24h  = end_window - pd.Timedelta(hours=24)
    df_24h = _ensure_dt_index(df_all.loc[start_24h:end_window])

    flow_lph = None
    try:
        t1h = end_window - pd.Timedelta(minutes=60)
        win = _ensure_dt_index(df_all.loc[t1h:end_window].copy())
        if not win.empty and "Supply_Consumption_Excl_Fills" in win.columns:
            per_min = win["Supply_Consumption_Excl_Fills"].resample("1min").sum().fillna(0)
            last60 = per_min.last("60min")
            if not last60.empty:
                flow_lph = float(last60.mean() * 60.0)
    except Exception:
        pass

    daily_fluid_use  = float(df_24h.get("Supply_Consumption_Excl_Fills", pd.Series(dtype=float)).sum())
    daily_ext_losses = float(df_24h.get("External_Losses", pd.Series(dtype=float)).sum())

    def _state_to_tone(v: Optional[int]) -> str:
        if v == 1: return "ok"
        if v == 2: return "warn"
        if v == 3: return "bad"
        return "neu"

    lp_state = int(df_view.get("LP_pump_state", pd.Series([1])).dropna().iloc[-1]) if "LP_pump_state" in df_view.columns and df_view["LP_pump_state"].notna().any() else 1
    hp_state = int(df_view.get("HP_pump_state", pd.Series([1])).dropna().iloc[-1]) if "HP_pump_state" in df_view.columns and df_view["HP_pump_state"].notna().any() else 1
    lp_tone, hp_tone = _state_to_tone(lp_state), _state_to_tone(hp_state)

    def freq_label(runs: Optional[float]) -> str:
        try:
            if runs and runs > 0: return f"every {int(1440 // float(runs))} min"
        except Exception: pass
        return "No runs"

    lp_runs_val = float(df_view["LP_Runs_24h"].dropna().iloc[-1]) if "LP_Runs_24h" in df_view and df_view["LP_Runs_24h"].notna().any() else None
    hp_runs_val = float(df_view["HP_pump_rate_2h"].dropna().iloc[-1])*12.0 if "HP_pump_rate_2h" in df_view and df_view["HP_pump_rate_2h"].notna().any() else None
    lp_freq_val, hp_freq_val = freq_label(lp_runs_val), freq_label(hp_runs_val)

    def mins_to_h(m): 
        try: return round(float(m)/60.0, 1)
        except Exception: return None

    lp_cum_h = mins_to_h(df_view.get("LP_Runtime_Cumulative_min", pd.Series(dtype=float)).dropna().iloc[-1]) if "LP_Runtime_Cumulative_min" in df_view and df_view["LP_Runtime_Cumulative_min"].notna().any() else None
    hp_cum_h = mins_to_h(df_view.get("HP_Runtime_Cumulative_min", pd.Series(dtype=float)).dropna().iloc[-1]) if "HP_Runtime_Cumulative_min" in df_view and df_view["HP_Runtime_Cumulative_min"].notna().any() else None

    with k1: kpi_card("Flow Rate (L/h)", "—" if flow_lph is None else f"{int(round(flow_lph)):,}", tone="neu", subtitle="last hour")
    with k2: kpi_card("Daily Fluid Use (L)", f"{int(round(daily_fluid_use)):,}", tone="ok", subtitle="Last 24 hours")
    with k3: kpi_card("Daily External Losses (L)", f"{int(round(daily_ext_losses)):,}", tone="ok", subtitle="Last 24 hours")
    with k4: kpi_card("LP Daily Run Frequency", lp_freq_val, tone=lp_tone, subtitle=(f"Cumulative Pump Runtime: {lp_cum_h} h" if lp_cum_h is not None else None))
    with k5: kpi_card("HP Daily Run Frequency", hp_freq_val, tone=hp_tone, subtitle=(f"Cumulative Pump Runtime: {hp_cum_h} h" if hp_cum_h is not None else None))

    def steady_state_percent(df: pd.DataFrame) -> Optional[float]:
        if "is_steady_state" not in df.columns: return None
        s = df["is_steady_state"].dropna()
        if s.empty: return None
        return float(s.astype(int).mean()*100.0)

    with k6:
        steady_pct = steady_state_percent(df_view)
        if steady_pct is not None:
            p = max(0, min(100, int(round(steady_pct))))
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-title">Steady State — selected period ({range_txt_sel})</div>
              <div style="width:38px; height:38px; border-radius:50%;
                          background:conic-gradient(#2ecc71 {p}%, #303030 {p}%);
                          display:flex; align-items:center; justify-content:center;
                          font-size:0.8rem; font-weight:800; color:#e8f0f6; margin:0 auto;">
                  {p}%
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            kpi_card("Steady State — selected period", "—")

    # condensed LP/HP grid (unchanged)
    def _last(col: str) -> Optional[float]:
        if col in df_view.columns and df_view[col].notna().any():
            return float(df_view[col].dropna().iloc[-1])
        return None

    with k7:
        def _sq(v: Optional[float], thr: float) -> str:
            cls = "square-red"
            if v is not None and v >= thr: cls = "square-green"
            val = "—" if v is None else f"{v:.0f} bar"
            return f'<span class="square {cls}" title="{val}"></span><span style="font-size:0.7rem; color:#cfe0ea;">{val}</span>'
        LPA, LPB, HPA, HPB = _last("HPU_LPA_OUT"), _last("HPU_LPB_OUT"), _last("HPU_HPA_OUT"), _last("HPU_HPB_OUT")
        st.markdown(f"""
        <div class="kpi-card kpi-neu" style="padding:6px 8px;">
          <div style="display:grid; grid-template-columns: 34px 1fr 1fr; gap:4px 10px; align-items:center;">
            <div></div><div style="font-size:0.72rem; color:#a8b4bf;">A</div><div style="font-size:0.72rem; color:#a8b4bf;">B</div>
            <div style="font-size:0.72rem; color:#a8b4bf;">LP</div> <div>{_sq(LPA,150.0)}</div> <div>{_sq(LPB,150.0)}</div>
            <div style="font-size:0.72rem; color:#a8b4bf;">HP</div> <div>{_sq(HPA,400.0)}</div> <div>{_sq(HPB,400.0)}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ----------------------------
    # Charts (unchanged visuals)
    # ----------------------------
    def _plotly_common_layout(fig: go.Figure, title: Optional[str], ytitle: str, height: int = 340,
                              x_start: Optional[pd.Timestamp]=None, x_end: Optional[pd.Timestamp]=None,
                              daily: bool=False, hovermode: str="x unified"):
        fig.update_layout(title_text=None)
        fig.update_layout(
            template="otter_darkgrey",
            yaxis_title=ytitle,
            hovermode=hovermode,
            height=height,
            margin=dict(l=10, r=10, t=10, b=28),
            uirevision="otter",
            paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
            xaxis=dict(
                gridcolor=GRID_COL, zerolinecolor=GRID_COL,
                rangeslider=dict(visible=True, bgcolor="#12151c"),
                rangeselector=dict(
                    x=0, xanchor="left",
                    buttons=[
                        dict(count=12, step="hour", stepmode="backward", label="12h"),
                        dict(count=1,  step="day",  stepmode="backward", label="1d"),
                        dict(count=7,  step="day",  stepmode="backward", label="7d"),
                        dict(count=30, step="day",  stepmode="backward", label="30d"),
                        dict(count=90, step="day",  stepmode="backward", label="90d"),
                        dict(count=1,  step="year", stepmode="backward", label="1y"),
                        dict(step="all", label="All"),
                    ]
                ),
            ),
            yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        )
        fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1)
        fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1)
        if x_start is not None and x_end is not None:
            fig.update_xaxes(range=[x_start, x_end])
        ht = "<b>%{x|%d-%b-%Y %H:%M}</b><br>%{y:.0f}<extra></extra>"
        if daily:
            ht = "<b>%{x|%d-%b-%Y}</b><br>%{y:.0f} L/day<extra></extra>"
        fig.update_traces(hovertemplate=ht)

    df_plot_full = downsample(df_calc, resample_rule)
    c1, c2, c3 = st.columns([1.25,1.25,1.25])

    def extract_valve_events(df: pd.DataFrame) -> pd.DataFrame:
        if "Valve_Event_Log" not in df.columns:
            return pd.DataFrame(columns=["time","text"])
        ev = df.loc[df["Valve_Event_Log"].notna(), ["Valve_Event_Log"]].copy()
        if ev.empty: return pd.DataFrame(columns=["time","text"])
        txt = ev["Valve_Event_Log"].astype(str)
        is_fill = txt.str.contains(r"\+\s*\d+\s*L", flags=re.IGNORECASE, regex=True) | txt.str.contains(r"\bfill(ed)?\b", flags=re.IGNORECASE, regex=True)
        is_valveish = txt.str.contains(r"(open|close|pmv|tdv|pdv|amv|valve)", flags=re.IGNORECASE, regex=True)
        keep = ~is_fill & is_valveish
        ev = ev[keep].copy()
        if ev.empty: return pd.DataFrame(columns=["time","text"])
        ev = ev.rename(columns={"Valve_Event_Log":"text"}); ev["time"] = ev.index
        return ev[["time","text"]].reset_index(drop=True)

    lp_valve_events = extract_valve_events(df_calc)

    with c1:
        st.caption("Tanks — Supply & Return")
        fig = go.Figure()
        if "HPU_SPLY_LEV_L" in df_plot_full.columns: fig.add_trace(go.Scatter(x=df_plot_full.index, y=df_plot_full["HPU_SPLY_LEV_L"], mode="lines", name="HPU_SPLY_LEV_L"))
        if "HPU_RET_LEV_L"  in df_plot_full.columns: fig.add_trace(go.Scatter(x=df_plot_full.index, y=df_plot_full["HPU_RET_LEV_L"],  mode="lines", name="HPU_RET_LEV_L"))
        _plotly_common_layout(fig, None, "Litres", 320, df_view.index.min(), df_view.index.max())
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    with c2:
        st.caption("LP — A/B Output, SCM1 LP Cons. (with valve markers)")
        fig = go.Figure()
        for col in ["HPU_LPA_OUT","HPU_LPB_OUT","SCM1_LP_CONS"]:
            if col in df_plot_full.columns:
                fig.add_trace(go.Scatter(x=df_plot_full.index, y=df_plot_full[col], mode="lines", name=col))
        lp_valve_events = extract_valve_events(df_calc)
        if not lp_valve_events.empty:
            base_col = None
            for c in ["HPU_LPA_OUT","HPU_LPB_OUT","SCM1_LP_CONS"]:
                if c in df_plot_full.columns and df_plot_full[c].notna().any():
                    base_col = c; break
            if base_col:
                base = df_plot_full[base_col].dropna().sort_index()
                xs, ys, texts = [], [], []
                for t, txt in zip(lp_valve_events["time"], lp_valve_events["text"]):
                    t = pd.to_datetime(t); idx = base.index.searchsorted(t, side="right") - 1
                    if idx >= 0:
                        xs.append(t); ys.append(float(base.iloc[idx])); texts.append(str(txt))
                if xs:
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode="markers", name="Valve event",
                        marker=dict(symbol="circle", size=8, color="#bbbbbb", line=dict(width=0.7, color="#222")),
                        text=texts, hovertemplate="<b>%{x|%d-%b-%Y %H:%M}</b><br>%{text}<extra></extra>", hoverlabel=dict(namelength=-1)
                    ))
        _plotly_common_layout(fig, None, "bar", 320, df_view.index.min(), df_view.index.max(), hovermode="closest")
        fig.update_traces(selector=dict(name="Valve event"), hovertemplate="<b>%{x|%d-%b-%Y %H:%M}</b><br>%{text}<extra></extra>", hoverlabel=dict(namelength=-1))
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    with c3:
        st.caption("Baseline Consumption – 12 h drops")
        fig = go.Figure()
        if "baseline_drop_L" in df_calc.columns and df_calc["baseline_drop_L"].dropna().size:
            drops = df_calc["baseline_drop_L"].dropna()
            pct95 = drops.quantile(0.95)
            clean = drops[(drops >= 0) & (drops <= pct95)]
            if not clean.empty:
                fig.add_trace(go.Scatter(x=clean.index, y=clean.values, mode="markers",
                    name="Filtered 12 h Drop",
                    marker=dict(size=7, color="#1f77b4", line=dict(width=0.6, color="#d0d8e0")),
                    hovertemplate="<b>%{x|%d-%b-%Y}</b><br>%{y:.1f} L<extra></extra>"
                ))
                med = float(clean.median())
                fig.add_hline(y=med, line_dash="dash", line_color="#77b0ff",
                              annotation_text=f"Median = {med:.1f} L", annotation_position="top left")
        _plotly_common_layout(fig, None, "Tank Level Drop (L)", 320, df_view.index.min(), df_view.index.max())
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    # Row 2 (left two identical; right one is state overlay now using HDBSCAN states)
    e1, e2, e3 = st.columns([1.25,1.25,1.25])

    with e1:
        st.caption("15-min Classification — Supply Excl. Fills with state overlay (HDBSCAN)")
        if model is None or xform is None or meta is None:
            st.info("HDBSCAN model not found — cannot render classification timeline.")
        else:
            start_cls = df_view.index.min()
            end_cls   = df_view.index.max()
            pred_15 = cluster_states_15min(df_full, model, xform, meta, start=start_cls, end=end_cls)

            supply_name = "Supply_Consumption_Excl_Fills"
            sup = df_full[[supply_name]].resample("15T").mean() if supply_name in df_full.columns else pd.DataFrame(index=pred_15.index)
            sup = sup.reindex(pred_15.index).fillna(0.0)

            fig = go.Figure()
            if not sup.empty:
                fig.add_trace(go.Scatter(x=sup.index, y=sup[supply_name], mode="lines", name=supply_name, line=dict(width=1.6)))

            if not pred_15.empty:
                for s_cls in sorted(pred_15["state"].dropna().unique().tolist()):
                    m = pred_15["state"].eq(s_cls)
                    fig.add_trace(go.Scatter(
                        x=pred_15.index[m],
                        y=sup.reindex(pred_15.index)[supply_name][m] if not sup.empty else np.zeros(m.sum()),
                        mode="markers", name=s_cls,
                        marker=dict(size=8, line=dict(width=0.6, color="#222")),
                        hovertemplate="<b>%{x|%d-%b-%Y %H:%M}</b><br>state=%{text}<br>Supply=%{y:.1f}<extra></extra>",
                        text=[s_cls]*int(m.sum())
                    ))

            _plotly_common_layout(fig, None, "L/15-min (mean)", 360, df_view.index.min(), df_view.index.max(), hovermode="closest")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    with e2:
        st.caption("Supply Consumption Breakdown — Daily Totals")
        def _empty_daily_index(df: pd.DataFrame) -> pd.DatetimeIndex:
            if df.empty or df.index.size == 0:
                return pd.DatetimeIndex([])
            idx = pd.to_datetime(df.index, errors="coerce")
            idx = idx[~idx.isna()]
            if idx.size == 0: return pd.DatetimeIndex([])
            return pd.date_range(idx.min().normalize(), idx.max().normalize(), freq="D")
        def _daily_sum(df: pd.DataFrame, col: str) -> pd.Series:
            df = _ensure_dt_index(df.copy())
            if col not in df.columns: return pd.Series(0.0, index=_empty_daily_index(df))
            s = df[col].dropna()
            if s.empty: return pd.Series(0.0, index=_empty_daily_index(df))
            return s.resample("D").sum().fillna(0.0)

        supply = _daily_sum(df_calc, "Supply_Consumption_Excl_Fills")
        fcv    = _daily_sum(df_calc, "FCV_Fluid_Usage")
        valves = _daily_sum(df_calc, "Valve_Operation_Fluid")
        umb    = _daily_sum(df_calc, "umbilical_charge_volume")

        idx = supply.index.union(fcv.index).union(valves.index).union(umb.index)
        supply = supply.reindex(idx, fill_value=0.0)
        fcv    = fcv.reindex(idx, fill_value=0.0)
        valves = valves.reindex(idx, fill_value=0.0)
        umb    = umb.reindex(idx, fill_value=0.0)
        unacc  = (supply - (fcv + valves + umb)).clip(lower=0.0)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=idx, y=fcv.values,    name="FCV Usage"))
        fig.add_trace(go.Bar(x=idx, y=valves.values, name="Valve Operation"))
        fig.add_trace(go.Bar(x=idx, y=umb.values,    name="Umbilical Charge"))
        fig.add_trace(go.Bar(x=idx, y=unacc.values,  name="Unaccounted"))
        fig.update_layout(barmode="stack")
        _plotly_common_layout(fig, None, "L/day", 360, df_view.index.min().normalize(), df_view.index.max().normalize(), daily=True)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    with e3:
        st.caption("HP — A/B Output, SCM1 HP Cons.")
        fig = go.Figure()
        for col in ["HPU_HPA_OUT","HPU_HPB_OUT","SCM1_HP_CONS"]:
            if col in df_plot_full.columns:
                fig.add_trace(go.Scatter(x=df_plot_full.index, y=df_plot_full[col], mode="lines", name=col))
        _plotly_common_layout(fig, None, "bar", 360, df_view.index.min(), df_view.index.max())
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    # ----------------------------
    # Logs + System Status (unchanged)
    # ----------------------------
    colL, colC = st.columns([1,2])

    with colL:
        st.caption(f"Daily Fluid Use — selected period ({range_txt_sel})")
        def _daily_sum2(df: pd.DataFrame, col: str) -> pd.Series:
            df = _ensure_dt_index(df.copy())
            if col not in df.columns: return pd.Series(dtype=float)
            s = df[col].dropna()
            if s.empty: return pd.Series(dtype=float)
            return s.resample("D").sum().fillna(0.0)
        daily_use = _daily_sum2(df_view, "Supply_Consumption_Excl_Fills")
        if not daily_use.empty:
            tbl = pd.DataFrame({"Date": daily_use.index.strftime("%d-%m-%Y"),
                                "Fluid Use (L)": daily_use.values.round(0).astype(int)})
            if HAVE_AGGRID:
                gb = GridOptionsBuilder.from_dataframe(tbl)
                gb.configure_pagination(paginationAutoPageSize=True)
                gb.configure_default_column(editable=False, groupable=False)
                AgGrid(tbl, gridOptions=gb.build(), theme="streamlit", height=220,
                       fit_columns_on_grid_load=True, update_mode=GridUpdateMode.NO_UPDATE)
            else:
                st.markdown("""
                <style>
                .compact-holder { display:inline-block; }
                .compact-table  { width:auto; table-layout:fixed; border-collapse:collapse; }
                .compact-table th, .compact-table td { padding: 4px 8px; border-bottom: 1px solid #303030; color:#cfe0ea; font-size: 0.85rem; line-height: 1.1; }
                .compact-table th { color:#d9e6f2; background:#000000; font-weight:600; }
                .compact-table td.date { white-space:nowrap; }
                .compact-table td.num-left { text-align:left; }
                </style>""", unsafe_allow_html=True)
                headers = "<tr><th>Date</th><th>Fluid Use (L)</th></tr>"
                rows = "".join([f"<tr><td class='date'>{d}</td><td class='num-left'>{v}</td></tr>" for d, v in zip(tbl["Date"], tbl["Fluid Use (L)"])])
                st.markdown(f"""
                <div class="compact-holder">
                  <div style="height:200px; overflow:auto; border:1px solid #303030; border-radius:8px;">
                    <table class="compact-table">
                      <thead>{headers}</thead>
                      <tbody>{rows}</tbody>
                    </table>
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No data to compute daily fluid use.")

        st.caption("System Status")
        def chip(ok: Optional[bool], label: str, warn: bool=False) -> str:
            if ok is None:
                return f'<span class="badge"><span class="dot" style="background:#789;"></span>{label}</span>'
            cls = "dot-ok" if ok and not warn else ("dot-warn" if warn and ok else "dot-bad")
            return f'<span class="badge"><span class="dot {cls}"></span>{label}</span>'
        def last_bool(col) -> Optional[bool]:
            if col in df_view.columns and df_view[col].notna().any():
                return bool(df_view[col].dropna().iloc[-1])
            return None
        chips_html = " ".join([
            chip(last_bool("is_steady_state"), "Steady State"),
            chip(last_bool("is_lp_low"), "LP Low"),
            chip(last_bool("is_hp_low"), "HP Low"),
            chip(last_bool("is_lp_high"), "LP High"),
            chip(last_bool("is_hp_high"), "HP High"),
            chip(last_bool("is_no_redundancy"), "Redundancy", warn=True),
            chip(last_bool("is_supply_tank_low"), "Supply Tank Low", warn=True),
            chip(last_bool("is_pressurising"), "Pressurising", warn=True),
            chip(last_bool("is_fcv_ops"), "FCV Operating", warn=True),
            chip(last_bool("is_losses_high"), "Losses High", warn=True),
        ])
        st.markdown(f'<div class="status-panel">{chips_html}</div>', unsafe_allow_html=True)

    with colC:
        st.caption(f"Event Log — selected period ({range_txt_sel})")
        log_max = 500
        if "Valve_Event_Log" in df_view.columns:
            log_df = df_view.loc[df_view["Valve_Event_Log"].notna(), ["Valve_Event_Log"]].tail(log_max).copy()
            log_df = _ensure_dt_index(log_df)
            if getattr(log_df.index, "tz", None):
                log_df.index = log_df.index.tz_localize(None)
            out = pd.DataFrame({"Date/Time": log_df.index.strftime("%d-%m-%Y %H:%M:%S"), "Event Log": log_df["Valve_Event_Log"].astype(str).values})
            if HAVE_AGGRID:
                gb = GridOptionsBuilder.from_dataframe(out)
                gb.configure_pagination(paginationAutoPageSize=True)
                gb.configure_default_column(editable=False, groupable=False, filter=True)
                AgGrid(out, gridOptions=gb.build(), theme="streamlit", height=240,
                       fit_columns_on_grid_load=True, update_mode=GridUpdateMode.NO_UPDATE)
            else:
                rows = "".join([f"<tr><td class='date'>{d}</td><td class='wrap'>{t}</td></tr>" for d,t in zip(out["Date/Time"], out["Event Log"])])
                st.markdown(f"""
                <style>
                .compact-table  {{ width:auto; table-layout:fixed; border-collapse:collapse; }}
                .compact-table th, .compact-table td {{ padding: 4px 8px; border-bottom: 1px solid #303030; color:#cfe0ea; font-size: 0.85rem; line-height: 1.1; }}
                .compact-table th {{ color:#d9e6f2; background:#000; font-weight:600; }}
                .compact-table td.date {{ white-space:nowrap; }}
                .compact-table td.wrap {{ white-space:normal; word-break:break-word; }}
                </style>
                <div style="height:220px; overflow:auto; border:1px solid #303030; border-radius:8px;">
                  <table class="compact-table">
                    <thead><tr><th>Date/Time</th><th>Event Log</th></tr></thead>
                    <tbody>{rows}</tbody>
                  </table>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No `Valve_Event_Log` column present.")

    # ----------------------------
    # Operator Quick Plot (unchanged)
    # ----------------------------
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.divider()
    qp_left, qp_right = st.columns([1,3])

    with qp_left:
        st.caption("Operator Quick Plot")
        df_plot_src = df_full
        st.markdown('<div class="oqp-scope">', unsafe_allow_html=True)
        with st.form("operator_quick_plot_form", clear_on_submit=False):
            all_tags = df_plot_src.columns.tolist()
            selected_signals = st.multiselect(
                "Signals",
                options=sorted(all_tags),
                default=[c for c in ["HPU_LPA_OUT","HPU_LPB_OUT","is_steady_state","steady_state"] if c in all_tags],
                max_selections=20,
                help="Choose tags to overlay"
            )
            tf_mode = st.radio("Timeframe", ["Use current window","Custom"], horizontal=True, index=0)
            if tf_mode == "Custom":
                d1, d2 = st.columns(2)
                _sd = d1.date_input("Start date", value=df_view.index.min().date())
                _ed = d2.date_input("End date",   value=df_view.index.max().date())
                t1, t2 = st.columns(2)
                _st = t1.time_input("Start time", value=pd.Timestamp(df_view.index.min()).time())
                _et = t2.time_input("End time",   value=pd.Timestamp(df_view.index.max()).time())
                qp_start = pd.Timestamp.combine(pd.to_datetime(_sd).date(), _st)
                qp_end   = pd.Timestamp.combine(pd.to_datetime(_ed).date(), _et)
            else:
                qp_start, qp_end = df_view.index.min(), df_view.index.max()
            st.caption(f"Window: {qp_start:%d-%m-%Y %H:%M} → {qp_end:%d-%m-%Y %H:%M} · Resample: {resample_rule}")
            create_plot = st.form_submit_button("Create Plot", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def _qp_prepare(df, start, end, resample_rule):
        _raw = _ensure_dt_index(df.loc[start:end].copy())
        return downsample(_raw, resample_rule)

    with qp_right:
        if not create_plot:
            st.info("Choose tags and press **Create Plot**.")
        else:
            if not selected_signals:
                st.warning("Select at least one signal.")
            else:
                with st.spinner("Building plot..."):
                    df_ds = _qp_prepare(df_plot_src, qp_start, qp_end, resample_rule)
                    primary_cols, binary_cols, skipped = [], [], []
                    for sig in selected_signals:
                        if sig not in df_ds.columns: continue
                        s = df_ds[sig].dropna()
                        if s.empty: continue
                        if pd.api.types.is_numeric_dtype(s):
                            if s.isin([0,1]).all() and s.nunique() <= 2:
                                binary_cols.append(sig)
                            else:
                                primary_cols.append(sig)
                        else:
                            skipped.append(sig)
                    fig = go.Figure()
                    for sig in primary_cols:
                        s = df_ds[sig].dropna()
                        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=sig, mode="lines",
                                                 hovertemplate="<b>%{x|%d-%b-%Y %H:%M}</b><br>%{y:.4g}<extra></extra>"))
                    for sig in binary_cols:
                        s = df_ds[sig].dropna()
                        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=f"{sig} (state)", mode="lines", yaxis="y2",
                                                 hovertemplate="<b>%{x|%d-%b-%Y %H:%M}</b><br>state=%{y:.0f}<extra></extra>"))
                    _plotly_common_layout(fig, None, "value", 360, qp_start, qp_end, daily=False, hovermode="x unified")
                    if binary_cols:
                        fig.update_layout(yaxis2=dict(title="state", overlaying="y", side="right", range=[-0.1,1.1],
                                                      tickmode="array", tickvals=[0,1], showgrid=False, zeroline=False))
                    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
                    if skipped: st.caption("Skipped non-numeric tags: " + ", ".join(skipped))


if __name__ == "__main__":
    main()
