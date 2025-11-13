# dashboard_cluster_app_v7.py
# Streamlit clustering dashboard — v20 functionality + improved AUTO cluster→state labelling
import streamlit as st

def _safe_set_page_config(*args, **kwargs):
    try:
        st.set_page_config(*args, **kwargs)
    except Exception:
        pass

def main():
    # =========================
    # Imports
    # =========================
    from pathlib import Path
    from typing import Dict, Tuple, List
    import json

    import numpy as np
    import pandas as pd
    import joblib

    import plotly.express as px
    import plotly.graph_objects as go

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from hdbscan import prediction as hdb_pred

    # =========================
    # Defaults (override in sidebar if needed)
    # =========================
    DEFAULT_DATA_PATH  = Path(r"C:\Users\rosst\OneDrive\Control Integrity\otter-hydraulic-system\data\df_all_otter_labeled_cleaned_v1_250825.parquet")
    DEFAULT_MODEL_DIR  = Path(r"C:\Users\rosst\OneDrive\Control Integrity\otter-hydraulic-system\notebooks\models")
    MODEL_FILENAME = "hdbscan_pca_model_v7.pkl"
    XFORM_FILENAME = "preprocess_scaler_pca_v7.pkl"   # dict: {"scaler": StandardScaler, "pca": PCA}
    META_FILENAME  = "hdbscan_meta_v7.json"           # feature_names, boolean_features, roll_windows, resample_rule, label_remap, optional cluster_to_state

    # Dashboard quick picks
    FEATURES_TO_PLOT = [
        "Supply_Consumption_Excl_Fills",
        "External_Losses_mean_30min",
        "LP_pump_rate_2h_mean_30min",
        "HP_pump_rate_2h_mean_30min",
        "FCV_Fluid_Usage",
        "LP_Px_Delta", "HP_Px_Delta", "SCM1_LP_CONS", "SCM1_HP_CONS",
    ]
    STEADY_LABELS = {"Steady State"}
    ROLLING_SUFFIXES = ("_mean_30min","_std_30min","_mean_1h","_std_1h","_mean_2h","_std_2h")

    # =========================
    # Theming
    # =========================
    PAPER_BG  = "#000000"
    PLOT_BG   = "#1a1a1a"
    GRID_COL  = "#333333"
    FONT_COL  = "#FFFFFF"

    def style_fig(fig, title=None, x_title=None, y_title=None, legend_title=None):
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
            font=dict(color=FONT_COL, size=13),
            title=dict(text=title or "", x=0.02, xanchor="left", y=0.98, font=dict(color=FONT_COL)),
            margin=dict(l=50, r=20, t=50, b=50),
            legend=dict(
                title=legend_title or "",
                font=dict(color=FONT_COL),
                bgcolor=PLOT_BG, bordercolor=PLOT_BG,
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0,
            ),
            hoverlabel=dict(bgcolor=PLOT_BG, font_color=FONT_COL),
        )
        fig.update_xaxes(title=x_title or "", gridcolor=GRID_COL, zeroline=False, showline=True, linecolor=GRID_COL)
        fig.update_yaxes(title=y_title or "", gridcolor=GRID_COL, zeroline=False, showline=True, linecolor=GRID_COL)
        return fig

    # =========================
    # Feature helpers (match your builder)
    # =========================
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

        if not parts:
            raise ValueError("No matching columns to downsample.")
        df_ds = pd.concat(parts, axis=1)
        return df_ds.loc[:, ~df_ds.columns.duplicated(keep="first")]

    def add_log1p_columns(df: pd.DataFrame, cols: list, prefix: str = "log1p_"):
        df = df.copy()
        for c in cols:
            if c in df.columns:
                df[f"{prefix}{c}"] = np.log1p(df[c].astype(float).clip(lower=-0.999))
        return df

    def add_rolling_features(df_ds: pd.DataFrame, base_numeric_cols: list, roll_windows: dict):
        cols = [c for c in base_numeric_cols if c in df_ds.columns]
        if not cols:
            return df_ds.copy()
        out_parts = [df_ds]
        for label, w in roll_windows.items():
            roll = df_ds[cols].rolling(window=w, min_periods=1)
            mean_df = roll.mean(); std_df = roll.std().fillna(0)
            mean_df.columns = [f"{c}_mean_{label}" for c in cols]
            std_df.columns  = [f"{c}_std_{label}"  for c in cols]
            out_parts.extend([mean_df, std_df])
        out = pd.concat(out_parts, axis=1)
        return out.copy()

    def build_feature_matrix(df_feat: pd.DataFrame, numeric_cols: list, boolean_cols: list):
        numeric_cols = [c for c in numeric_cols if c in df_feat.columns]
        boolean_cols = [c for c in boolean_cols if c in df_feat.columns]
        rolling_cols = [c for c in df_feat.columns if any(s in c for s in ROLLING_SUFFIXES)]
        feature_cols = sorted(list(dict.fromkeys(numeric_cols + boolean_cols + rolling_cols)))
        X = df_feat[feature_cols].copy().fillna(0)
        return X, feature_cols

    def extract_base_feature_lists(meta: dict, df_raw: pd.DataFrame):
        feat_order = meta.get("feature_names", [])
        bool_feats = set(meta.get("boolean_features", []))
        base_numeric = [
            c for c in feat_order
            if (c in df_raw.columns)
            and (c not in bool_feats)
            and (not c.startswith("log1p_"))
            and (not c.endswith(ROLLING_SUFFIXES))
            and _is_numeric_series(df_raw[c])
        ]
        base_boolean = [c for c in bool_feats if c in df_raw.columns]
        base_numeric = list(dict.fromkeys(base_numeric))
        base_boolean = list(dict.fromkeys(base_boolean))
        return base_numeric, base_boolean

    # =========================
    # Dynamic thresholds + AUTO LABEL (improved)
    # =========================
    def _derive_dynamic_thresholds(summary: pd.DataFrame) -> Dict[str, float]:
        """
        Learn soft thresholds from the current data window:
        - Use quantiles to adapt to drift but clamp to safe ranges.
        """
        q = summary.quantile
        # Loss activity thresholds from pump-rate distributions
        hp_thr = float(np.clip(q(0.75)["HP_pump_rate_2h_mean"], 0.7, 2.5)) if "HP_pump_rate_2h_mean" in summary else 1.1
        lp_thr = float(np.clip(q(0.75)["LP_pump_rate_2h_mean"], 0.6, 2.5)) if "LP_pump_rate_2h_mean" in summary else 1.0
        # “High losses” guard from losses flag prevalence
        losses_hi = float(np.clip(q(0.75)["is_losses_high_pct"], 25.0, 70.0)) if "is_losses_high_pct" in summary else 40.0
        # FCV dominance from fcv prevalence
        fcv_dominant = float(np.clip(q(0.90)["is_fcv_ops_pct"], 50.0, 95.0)) if "is_fcv_ops_pct" in summary else 75.0
        # Very-steady from steady prevalence
        steady_hi = float(np.clip(q(0.75)["is_steady_state_pct"], 80.0, 99.9)) if "is_steady_state_pct" in summary else 90.0

        return {
            "hp_pump_rate_min": hp_thr,
            "lp_pump_rate_min": lp_thr,
            "losses_pct_min":   losses_hi,
            "fcv_ops_pct_min":  fcv_dominant,
            "steady_state_pct_hi": steady_hi,
            "steady_state_pct_strict": 100.0,
            # misc
            "probability_floor": 0.70,
            "min_nonnoise_points": 300,
            "steady_downweight_loss": 5.0,
        }

    NOISE_RULES: Dict[str, object] = {
        "purity_min": 0.70,  # if no regime is strong -> impure
        "purity_keys": ["is_steady_state_pct", "is_fcv_ops_pct", "is_losses_high_pct"],
        # Mixed-bag signature (as you described)
        "mixed_steady_low": 40.0,
        "mixed_steady_high": 70.0,
        "mixed_fcv_min": 8.0,
        "mixed_losses_max": 30.0,
        # Reliability guards
        "mean_prob_min": 0.70,
        "min_points": 500,
    }

    def _cluster_summary(df_pred: pd.DataFrame) -> pd.DataFrame:
        cols_present = set(df_pred.columns)
        need_flags = ["is_fcv_ops","is_tank_fill","is_steady_state","is_steady_state_strict","is_losses_high"]
        have_flags = [c for c in need_flags if c in cols_present]
        agg = {"cluster": "size", "cluster_probability": "mean"}
        if "HP_pump_rate_2h" in cols_present: agg["HP_pump_rate_2h"] = "mean"
        if "LP_pump_rate_2h" in cols_present: agg["LP_pump_rate_2h"] = "mean"
        g = df_pred.groupby("cluster").agg(agg).rename(columns={
            "cluster": "n_points",
            "cluster_probability": "mean_prob",
            "HP_pump_rate_2h": "HP_pump_rate_2h_mean",
            "LP_pump_rate_2h": "LP_pump_rate_2h_mean",
        })
        for c in have_flags:
            g[f"{c}_pct"] = 100.0 * df_pred.groupby("cluster")[c].mean()
        # make sure required columns exist
        for k in ["is_fcv_ops_pct","is_tank_fill_pct","is_steady_state_pct","is_steady_state_strict_pct","is_losses_high_pct"]:
            if k not in g.columns: g[k] = 0.0
        if "HP_pump_rate_2h_mean" not in g.columns: g["HP_pump_rate_2h_mean"] = 0.0
        if "LP_pump_rate_2h_mean" not in g.columns: g["LP_pump_rate_2h_mean"] = 0.0
        g = g.fillna(0.0)
        g.insert(0, "cluster_id", g.index.astype(int))
        return g.reset_index(drop=True)

    def _noise_like(r: pd.Series, rules=NOISE_RULES) -> bool:
        if int(r["cluster_id"]) == -1:
            return True
        purity_vals = [float(r.get(k, 0.0))/100.0 for k in rules["purity_keys"] if k in r.index]
        max_purity = max(purity_vals) if purity_vals else 0.0
        low_purity = (max_purity < float(rules["purity_min"]))
        steady = float(r.get("is_steady_state_pct", 0.0))
        fcv    = float(r.get("is_fcv_ops_pct", 0.0))
        losses = float(r.get("is_losses_high_pct", 0.0))
        mixed  = (float(rules["mixed_steady_low"]) <= steady <= float(rules["mixed_steady_high"])) \
                 and (fcv >= float(rules["mixed_fcv_min"])) and (losses <= float(rules["mixed_losses_max"]))
        low_prob  = float(r.get("mean_prob", 0.0)) < float(rules["mean_prob_min"])
        too_small = int(r.get("n_points", 0)) < int(rules["min_points"])
        return (low_purity and (low_prob or too_small)) or (mixed and (low_prob or too_small))

    def _score_and_label(summary: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, str]]:
        # dynamic thresholds
        P = _derive_dynamic_thresholds(summary.copy())
        s = summary.copy()

        def sys_dep(r):
            # strict steady + pumps near off
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
            if cid == -1 or _noise_like(r):
                labels.append("Noise"); confs.append(1.0 if cid == -1 else 0.8); continue
            cand = {
                "System Depressurised": float(r["score_sys_depr"]),
                "FCV Operation"       : float(r["score_fcv"]),
                "HP Losses"           : float(r["score_hp_loss"]),
                "LP Losses"           : float(r["score_lp_loss"]),
                "Steady State"        : float(r["score_steady"]),
            }
            best_label = max(cand, key=cand.get); best_score = cand[best_label]
            if best_score <= 0.0:
                best_label = "Steady State" if r["is_steady_state_pct"] >= P["steady_state_pct_hi"] else "Other"
                best_score = 0.3 if best_label=="Steady State" else 0.1
            conf = float(np.clip(best_score, 0, 1))
            conf += 0.05*(r["mean_prob"] >= P["probability_floor"])
            conf += 0.05*(r["n_points"] >= P["min_nonnoise_points"])
            conf = float(np.clip(conf, 0, 1))
            labels.append(best_label); confs.append(round(conf,3))
        s["auto_state"] = labels; s["auto_conf"] = confs
        mapping = {int(r.cluster_id): r.auto_state for r in s.itertuples()}
        return s, mapping

    def auto_mapping_from_pred(pred: pd.DataFrame) -> Tuple[Dict[int,str], pd.DataFrame]:
        summary = _cluster_summary(pred)
        scored, mapping = _score_and_label(summary)
        return mapping, scored

    # =========================
    # Cache loaders
    # =========================
    @st.cache_resource(show_spinner=False)
    def load_artifacts_cached(model_path: str, xform_path: str, meta_path: str, m_mtime: float, x_mtime: float, meta_mtime: float):
        model = joblib.load(model_path)
        xform = joblib.load(xform_path)
        meta  = json.loads(Path(meta_path).read_text())
        return model, xform, meta

    @st.cache_resource(show_spinner=False)
    def load_data_cached(data_path: str, d_mtime: float):
        p = Path(data_path)
        if not p.exists():
            raise FileNotFoundError(f"DATA_PATH not found: {p}")
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        # Ensure DatetimeIndex (fallback to 'timestamp' column if present)
        if not isinstance(df.index, pd.DatetimeIndex):
            tcol = "timestamp" if "timestamp" in df.columns else None
            if tcol is None:
                raise TypeError("Expected a DatetimeIndex or a 'timestamp' column.")
            df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
            df = df.set_index(tcol)
        return df.sort_index()

    # =========================
    # Inference
    # =========================
    def predict_states(df_raw: pd.DataFrame, model, xform: dict, meta: dict, years_of_history: int = 1,
                       force_auto: bool = False):
        years = int(max(1, min(4, years_of_history)))
        if len(df_raw):
            cutoff_date = df_raw.index.max() - pd.DateOffset(years=years)
            df_raw = df_raw.loc[df_raw.index >= cutoff_date].copy()

        base_numeric_raw, base_boolean_raw = extract_base_feature_lists(meta, df_raw)
        
        missing_numeric = [c for c in base_numeric_raw if c not in df_raw.columns]
        missing_boolean = [c for c in base_boolean_raw if c not in df_raw.columns]

        if missing_numeric or missing_boolean:
            st.warning(
                f"Downsample WARN: {len(missing_numeric)} numeric + {len(missing_boolean)} boolean "
                f"features not in data.\nFirst few missing numeric: {missing_numeric[:10]}\n"
                f"First few missing boolean: {missing_boolean[:10]}"
            )
    
        df_15 = downsample_15min(
            df_raw,
            numeric_cols=base_numeric_raw,
            boolean_cols=base_boolean_raw,
            rule=meta.get("resample_rule", "15T"),
        )

        log_bases = [c for c in meta.get("log1p_features", []) if c in df_15.columns]
        df_15 = add_log1p_columns(df_15, log_bases)

        base_numeric_for_roll = [c for c in base_numeric_raw if c in df_15.columns]
        base_numeric_for_roll += [f"log1p_{c}" for c in log_bases if f"log1p_{c}" in df_15.columns]
        df_15_feat = add_rolling_features(df_15, base_numeric_for_roll, roll_windows=meta.get("roll_windows", {"30min":2,"1h":4,"2h":8}))

        feat_order = meta.get("feature_names", [])
        bool_feats = meta.get("boolean_features", [])
        X_all, _ = build_feature_matrix(df_15_feat, numeric_cols=feat_order, boolean_cols=bool_feats)
        X_df = X_all.reindex(columns=feat_order).fillna(meta.get("fillna_value", 0))

        scaler: StandardScaler = xform["scaler"]
        pca: PCA = xform["pca"]
        X_scaled = scaler.transform(X_df)
        X_pca    = pca.transform(X_scaled)

        labels_raw, strengths = hdb_pred.approximate_predict(model, X_pca)
        remap = meta.get("label_remap", {})
        labels = np.array([remap.get(int(x), int(x)) for x in labels_raw], dtype=int)

        out = df_15_feat.copy()
        out["cluster"] = labels
        out["cluster_probability"] = strengths

        # Prefer saved mapping unless force_auto
        c2s = None
        mapping_src = "model/meta"
        if hasattr(model, "cluster_to_state_") and not force_auto:
            c2s = {int(k): str(v) for k, v in getattr(model, "cluster_to_state_").items()}
        elif meta.get("cluster_to_state") and not force_auto:
            c2s = {int(k): str(v) for k, v in meta["cluster_to_state"].items()}

        scored = None
        if c2s is None or force_auto:
            mapping_src = "auto"
            c2s, scored = auto_mapping_from_pred(out)

        out["state"] = out["cluster"].map(c2s).fillna("Other")
        out["cluster_label"] = [f"{int(c)} · {s}" for c, s in zip(out["cluster"], out["state"])]

        return out, c2s, mapping_src, scored

    # =========================
    # UI — page styling
    # =========================
    _safe_set_page_config(page_title="Hydraulic Regime Clustering", layout="wide")
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color:{PAPER_BG}; }}
        [data-testid="stSidebar"] {{ background-color: {PAPER_BG}; }}
        [data-testid="stSidebar"] * {{ color: {FONT_COL} !important; }}
        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background-color: #111111 !important;
            color: {FONT_COL} !important;
            border-color: #333333 !important;
        }}
        [data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea {{
            background-color: #111111 !important;
            color: {FONT_COL} !important;
            border-color: #333333 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Hydraulic Regime Clustering & Status Prediction (15-min)")

    # =========================
    # Sidebar
    # =========================
    with st.sidebar:
        st.header("Options")
        years = st.slider("Years of history (1–4)", 1, 4, 1, 1, help="Filter to the most recent N years.")
        force_auto = st.checkbox("Force AUTO labelling (ignore saved mapping)", value=False)

        st.subheader("Paths (override if needed)")
        data_path_in  = Path(st.text_input("Data path",  str(DEFAULT_DATA_PATH)))
        model_dir_in  = Path(st.text_input("Model folder", str(DEFAULT_MODEL_DIR)))
        model_path_in = model_dir_in / MODEL_FILENAME
        xform_path_in = model_dir_in / XFORM_FILENAME
        meta_path_in  = model_dir_in / META_FILENAME

    # =========================
    # Load artifacts & data
    # =========================
    try:
        model, xform, meta = load_artifacts_cached(
            str(model_path_in), str(xform_path_in), str(meta_path_in),
            model_path_in.stat().st_mtime, xform_path_in.stat().st_mtime, meta_path_in.stat().st_mtime
        )
        df_raw = load_data_cached(str(data_path_in), data_path_in.stat().st_mtime)
    except Exception as e:
        st.error(f"Failed to load: {e}")
        st.stop()

    st.caption(f"Using data: `{data_path_in}`")
    st.write(f"Raw shape: **{df_raw.shape}**  |  Full range: **{df_raw.index.min()} – {df_raw.index.max()}**")

    # =========================
    # Predict
    # =========================
    pred, cluster_to_state, mapping_src, scored = predict_states(
        df_raw, model, xform, meta, years_of_history=years, force_auto=force_auto
    )
    used_start = pred.index.min() if isinstance(pred.index, pd.DatetimeIndex) else None
    used_end   = pred.index.max() if isinstance(pred.index, pd.DatetimeIndex) else None
    st.success(f"Predicted **{len(pred):,}** 15-minute rows • Window: **{used_start} – {used_end}**")

    if mapping_src == "auto":
        st.warning("Using *AUTO* cluster→state mapping (no saved mapping found or override enabled).")
        if scored is not None:
            with st.expander("Auto-labelled cluster summary"):
                st.dataframe(
                    scored[[
                        "cluster_id","n_points","mean_prob",
                        "is_fcv_ops_pct","is_steady_state_pct","is_losses_high_pct",
                        "HP_pump_rate_2h_mean","LP_pump_rate_2h_mean",
                        "auto_state","auto_conf"
                    ]].sort_values("n_points", ascending=False),
                    use_container_width=True
                )

    # =========================
    # KPIs
    # =========================
    steady_mask = pred["state"].isin(STEADY_LABELS)
    steady_pct  = (100.0 * steady_mask.mean()) if len(pred) else 0.0
    current_state = pred["state"].iloc[-1] if len(pred) else "—"

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Steady-state % (last {years} yr{'s' if years>1 else ''})", f"{steady_pct:.1f}%")
    c2.metric("Current status (last 15 min)", current_state)
    c3.metric("Distinct states", f"{pred['state'].nunique()}")

    # =========================
    # Cluster frequency (full width, show ALL clusters)
    # =========================
    st.subheader("Cluster frequency")
    EXPECTED_CLUSTER_IDS = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    complete = pd.DataFrame({"cluster": EXPECTED_CLUSTER_IDS})
    complete["state"] = complete["cluster"].map(cluster_to_state).fillna("Other")
    complete["cluster_label"] = complete.apply(lambda r: f"{int(r['cluster'])} · {r['state']}", axis=1)

    counts_actual = (
        pred.groupby(["cluster"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
    )
    counts = complete.merge(counts_actual, on="cluster", how="left").fillna({"count": 0})
    counts["count"] = counts["count"].astype(int)

    fig = px.bar(counts, x="cluster_label", y="count")
    fig = style_fig(fig, y_title="count", x_title="cluster (id · state)")
    fig.update_xaxes(tickangle=-30, categoryorder="array", categoryarray=counts["cluster_label"])
    st.plotly_chart(fig, use_container_width=True, key="cluster_freq_all")

    # =========================
    # Time series (hourly median + state markers)
    # =========================
    st.subheader("Time series (select a feature)")
    avail_feats = [c for c in FEATURES_TO_PLOT if c in pred.columns]
    if not avail_feats:
        st.info("None of FEATURES_TO_PLOT found in engineered frame.")
    else:
        feat = st.selectbox("Feature", options=avail_feats, index=0, key="ts_feat")
        dfp = pred[[feat, "state"]].copy()

        def mode_or_na(s):
            s = s.dropna()
            return s.mode().iat[0] if len(s) else np.nan

        df_1h = pd.DataFrame({
            feat: dfp[feat].resample("1H").median(),
            "state": dfp["state"].resample("1H").agg(mode_or_na),
        }).dropna(subset=[feat])

        df_1h["smooth"] = df_1h[feat].rolling(7, center=True, min_periods=1).median()

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=df_1h.index, y=df_1h["smooth"], mode="lines",
                                    name=f"{feat} (1h median, 7-pt)", line=dict(width=1.6)))
        step = max(1, len(df_1h)//2000)
        for st_name in sorted(df_1h["state"].dropna().unique().tolist()):
            m = df_1h["state"].eq(st_name)
            fig_ts.add_trace(go.Scatter(
                x=df_1h.index[m][::step],
                y=df_1h[feat][m][::step],
                mode="markers",
                name=str(st_name),
                marker=dict(size=5, opacity=0.6),
            ))
        fig_ts = style_fig(fig_ts, title=f"{feat} (1h median + state markers)",
                           x_title="timestamp", y_title=feat, legend_title="state")
        fig_ts.update_layout(height=380)
        st.plotly_chart(fig_ts, use_container_width=True, key="timeseries_main")

    # =========================
    # Distributions (By state & By cluster)
    # =========================
    st.subheader("Distributions")
    fsel = st.multiselect("Choose features to compare", options=avail_feats, default=avail_feats[:4] if avail_feats else [], key="dist_feats")
    tab_state, tab_cluster = st.tabs(["By state", "By cluster"])

    if fsel:
        melt_cols = fsel + ["cluster","cluster_label","state"]
        melted = pred[melt_cols].reset_index(drop=True).melt(
            id_vars=["cluster","cluster_label","state"],
            value_vars=fsel,
            var_name="feature",
            value_name="value"
        )
        with tab_state:
            fig_box2 = px.box(melted, x="state", y="value", color="feature", points=False)
            fig_box2 = style_fig(fig_box2, title="Distribution by state", x_title="state", y_title="value", legend_title="feature")
            fig_box2.update_xaxes(categoryorder="total descending")
            st.plotly_chart(fig_box2, use_container_width=True, key="dist_by_state")

        with tab_cluster:
            order = (
                melted[["cluster","cluster_label"]]
                .drop_duplicates()
                .sort_values("cluster")["cluster_label"].tolist()
            )
            fig_box1 = px.box(melted, x="cluster_label", y="value", color="feature", points=False)
            fig_box1 = style_fig(fig_box1, title="Distribution by cluster", x_title="cluster (id · state)", y_title="value", legend_title="feature")
            fig_box1.update_xaxes(categoryorder="array", categoryarray=order, tickangle=-30)
            st.plotly_chart(fig_box1, use_container_width=True, key="dist_by_cluster")

    # =========================
    # Quick 2×2 feature scatters (with filters)
    # =========================
    st.subheader("Quick 2×2 feature scatters")
    state_opts = sorted(pred["state"].dropna().unique().tolist())
    cluster_opts = (
        pred[["cluster","cluster_label"]]
        .drop_duplicates()
        .sort_values("cluster")["cluster_label"].tolist()
    )
    cflt1, cflt2 = st.columns([1,1])
    with cflt1:
        sel_states = st.multiselect("States to include", options=state_opts, default=state_opts, key="qs_states")
    with cflt2:
        sel_clusters = st.multiselect("Clusters to include", options=cluster_opts, default=cluster_opts, key="qs_clusters")

    mask = pred["state"].isin(sel_states) & pred["cluster_label"].isin(sel_clusters)
    pred_view = pred.loc[mask].copy()

    pairs_default = [
        ("Supply_Consumption_Excl_Fills", "External_Losses_mean_30min"),
        ("Supply_Consumption_Excl_Fills", "LP_pump_rate_2h_mean_30min"),
        ("Supply_Consumption_Excl_Fills", "HP_pump_rate_2h_mean_30min"),
        ("Supply_Consumption_Excl_Fills", "FCV_Fluid_Usage"),
    ]
    opts = [c for c in FEATURES_TO_PLOT if c in pred_view.columns]

    def pick(fallback):
        a, b = fallback
        ax = a if a in opts else (opts[0] if opts else a)
        by = b if b in opts else (opts[1] if len(opts) > 1 else (opts[0] if opts else b))
        return ax, by

    x1, y1 = pick(pairs_default[0])
    x2, y2 = pick(pairs_default[1])
    x3, y3 = pick(pairs_default[2])
    x4, y4 = pick(pairs_default[3])

    pt_size  = st.slider("Point size", 3, 12, 6, key="qs_size")
    pt_alpha = st.slider("Point opacity", 10, 100, 85, step=5, key="qs_alpha") / 100.0

    def small_scatter(df, x, y, title, k):
        fig = px.scatter(df.reset_index(drop=True), x=x, y=y, color="state",
                         hover_data=["cluster","cluster_probability","cluster_label"])
        fig = style_fig(fig, title=title, x_title=x, y_title=y, legend_title="state")
        fig.update_traces(marker={"size": pt_size, "opacity": pt_alpha})
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True, key=k)

    c1, c2 = st.columns(2)
    small_scatter(pred_view, x1, y1, f"{x1} vs {y1}", "qs_plot_1")
    small_scatter(pred_view, x2, y2, f"{x2} vs {y2}", "qs_plot_2")
    c3, c4 = st.columns(2)
    small_scatter(pred_view, x3, y3, f"{x3} vs {y3}", "qs_plot_3")
    small_scatter(pred_view, x4, y4, f"{x4} vs {y4}", "qs_plot_4")

    st.caption("Tip: click legend entries to hide/show states; double-click to isolate one.")

    # =========================
    # Embedding view (3D UMAP→PCA)
    # =========================
    st.subheader("Embedding view (3D UMAP → PCA fallback)")
    embed_cols = [c for c in meta.get("feature_names", []) if c in pred.columns]
    Xn = pred[embed_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(Xn) > 10:
        try:
            import umap
            red3 = umap.UMAP(n_components=3, random_state=42)
            Z = red3.fit_transform(Xn.values)
            _title = "UMAP (3D)"
        except Exception:
            red3 = PCA(n_components=3, random_state=42)
            Z = red3.fit_transform(Xn.values)
            _title = "PCA (3D)"

        dfZ = pd.DataFrame(Z, columns=["Dim1","Dim2","Dim3"], index=Xn.index)
        dfZ["state"]   = pred.loc[dfZ.index, "state"]
        dfZ["cluster"] = pred.loc[dfZ.index, "cluster"].astype(int)
        fig3d = px.scatter_3d(dfZ, x="Dim1", y="Dim2", z="Dim3", color="state", hover_data=["cluster"], opacity=0.9)
        fig3d.update_traces(marker=dict(size=3))
        fig3d.update_layout(
            template="plotly_dark", title=_title, height=600,
            margin=dict(l=0,r=0,t=52,b=0),
            paper_bgcolor="#000000", plot_bgcolor="#000000",
            legend=dict(title=dict(text="state", font=dict(color="#FFFFFF")),
                        font=dict(color="#FFFFFF"), bgcolor="#000000",
                        orientation="h", y=1.02, x=0.0, xanchor="left"),
            scene=dict(
                bgcolor="#000000",
                xaxis=dict(showbackground=True, backgroundcolor="#0f0f0f", gridcolor="#333333"),
                yaxis=dict(showbackground=True, backgroundcolor="#0f0f0f", gridcolor="#333333"),
                zaxis=dict(showbackground=True, backgroundcolor="#0f0f0f", gridcolor="#333333"),
                aspectmode="cube",
            ),
            scene_camera=dict(eye=dict(x=1.7,y=1.7,z=1.25)),
        )
        st.plotly_chart(fig3d, use_container_width=True, key="embed_3d")
    else:
        st.info("Not enough points for 3D embedding.")

    # =========================
    # Mapping & download
    # =========================
    with st.expander("Cluster → State mapping (used)"):
        st.json(cluster_to_state)

    st.download_button(
        "Download mapping (JSON)",
        json.dumps(cluster_to_state, indent=2).encode("utf-8"),
        file_name="cluster_to_state_used.json",
        mime="application/json",
    )

    st.download_button(
        "Download predictions (CSV)",
        pred.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="pred_15min_clusters.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()
