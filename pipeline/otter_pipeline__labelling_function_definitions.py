from __future__ import annotations
import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
from IPython.display import HTML, display
import matplotlib.dates as mdates
from scipy.stats import pearsonr, spearmanr
from scipy.fft import fft, fftfreq
from sklearn.linear_model import LinearRegression
import sys
import os
from scipy.stats import pearsonr, spearmanr
from typing import Optional, Tuple, Dict, Any, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple, Union, List

########################################################################################
################################################################################################################################################

def add_low_fluid_consumption_label(
    df: pd.DataFrame,
    *,
    col: str = "Supply_Consumption_Excl_Fills",   # per-minute series (L/min)
    mode: str = "episode",                        # "episode" (hysteresis) or "sliding"
    # --- Option A (episode + hysteresis) ---
    enter_lpm: float = 52.0/60.0,                 # 1.167 L/min
    exit_lpm: float  = 62.0/60.0,                 # 1.333 L/min
    min_enter_min: int = 30,                      # ≥30 consecutive mins ≤ enter_lpm to start
    min_exit_min:  int = 3,                       # ≥3 consecutive mins > exit_lpm to end
    # --- Option B (sliding window stability) ---
    w_min: int = 45,                              # rolling window size (minutes)
    std_max_lpm: float = 0.08,                    # L/min std upper bound inside window
    slope_max_lpm_per_hr: float = 0.4,            # |Δ(L/min)| per hour inside window
    min_stable_min: int = 30,                     # enforce minimum True run length
    bridge_gap_min: int = 3,                      # fill ≤3-min False gaps in True runs
    # --- Output ---
    out_col: str = "is_consumption_low",          # boolean column added to df
    return_mask_only: bool = False                # if True, return (mask, meta) instead of df
):
    """
    Adds a boolean column marking 'low fluid consumption' spans using either:
      - mode='episode'  : hysteresis around (enter_lpm, exit_lpm) with min durations
      - mode='sliding'  : stability within a rolling window (low std and small slope)

    Assumes `df[col]` is per-minute L/min. If your index isn’t DatetimeIndex/minutely,
    the logic still works, but 'minute' thresholds should reflect your sampling rate.
    """
    d = df.copy()
    # hygiene
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors="coerce")
    d = d.sort_index()

    if col not in d.columns:
        raise KeyError(f"Column '{col}' not found.")

    s = pd.to_numeric(d[col], errors="coerce")

    # -------- Option A: episode + hysteresis --------
    def _label_episode(series: pd.Series) -> pd.Series:
        vals = series.values.astype(float)
        n = len(vals)
        out = np.zeros(n, dtype=bool)
        in_stable = False
        below_cnt = 0
        exit_cnt = 0
        for i, v in enumerate(vals):
            if np.isnan(v):
                # nan breaks enter streak; count towards exit if already in stable
                below_cnt = 0
                if in_stable:
                    exit_cnt += 1
                    if exit_cnt >= min_exit_min:
                        in_stable = False
                        exit_cnt = 0
                continue

            if not in_stable:
                if v <= enter_lpm:
                    below_cnt += 1
                    if below_cnt >= min_enter_min:
                        in_stable = True
                        start = i - min_enter_min + 1
                        out[start:i+1] = True
                        exit_cnt = 0
                else:
                    below_cnt = 0
            else:
                out[i] = True
                if v > exit_lpm:
                    exit_cnt += 1
                    if exit_cnt >= min_exit_min:
                        in_stable = False
                        exit_cnt = 0
                else:
                    exit_cnt = 0
        return pd.Series(out, index=series.index, dtype=bool)

    # -------- Option B: sliding window stability --------
    def _rolling_slope_per_hour(y_window: np.ndarray) -> float:
        # slope units: (L/min)/min; *60 => L/min per hour
        if np.isnan(y_window).any():
            return np.nan
        x = np.arange(len(y_window), dtype=float)
        x -= x.mean()
        y = y_window - np.nanmean(y_window)
        denom = (x**2).sum()
        if denom == 0:
            return 0.0
        m_per_min = float((x*y).sum() / denom)
        return m_per_min * 60.0

    def _enforce_min_and_bridge(mask: pd.Series, min_true=1, bridge=0) -> pd.Series:
        m = mask.fillna(False).copy()
        # bridge small False gaps
        if bridge and bridge > 0:
            g = (m != m.shift()).cumsum()
            for _, grp in m.groupby(g):
                if (grp.iloc[0] is False) and (len(grp) <= bridge):
                    m.loc[grp.index] = True
        # drop short True runs
        if min_true and min_true > 1:
            g2 = (m != m.shift()).cumsum()
            for _, grp in m.groupby(g2):
                if (grp.iloc[0] is True) and (len(grp) < min_true):
                    m.loc[grp.index] = False
        return m

    def _label_sliding(series: pd.Series) -> pd.Series:
        y = pd.to_numeric(series, errors="coerce")
        # rolling std (L/min)
        rstd = y.rolling(window=w_min, min_periods=w_min).std()
        # rolling slope per hour over window
        rslope = y.rolling(window=w_min, min_periods=w_min).apply(
            lambda w: _rolling_slope_per_hour(w.values), raw=False
        )
        base = (rstd <= std_max_lpm) & (rslope.abs() <= slope_max_lpm_per_hr)
        return _enforce_min_and_bridge(base, min_true=min_stable_min, bridge=bridge_gap_min)

    if mode.lower() == "episode":
        mask = _label_episode(s)
    elif mode.lower() == "sliding":
        mask = _label_sliding(s)
    else:
        raise ValueError("mode must be 'episode' or 'sliding'")

    # attach
    d[out_col] = mask.astype(bool)

    # small summary (can be handy for logs)
    groups = (mask != mask.shift()).cumsum()
    n_events = int(sum(bool(grp.iloc[0]) for _, grp in mask.groupby(groups)))
    meta = {
        "col": col,
        "mode": mode,
        "pct_true": float(mask.mean() * 100.0),
        "n_events": n_events
    }

    if return_mask_only:
        return mask.astype(bool), meta
    return d

##################################################################################################################################

def detect_umbilical_charge_events(
    df: pd.DataFrame,
    channel_thresholds: dict,
    level_col: str = 'HPU_SPLY_LEV_L',
    window: pd.Timedelta = pd.Timedelta('3H'),
    min_fluid: float = 5.0
) -> pd.DataFrame:
    """
    Detect umbilical charge events based on low→high transitions in either:
      - SCM1 consumption values (LP/HP)
      - HPU line outputs (LPB/HPB)
    
    Flags event if a signal crosses low→high threshold.
    """
    events = []

    for ch, (low_thr, high_thr) in channel_thresholds.items():
        s = df[ch].dropna().sort_index()
        above_low = s > low_thr
        rises = above_low & (~above_low.shift(fill_value=False))  # rising edges
        
        for t0 in rises[rises].index:
            slice_ = s.loc[t0 : t0 + window]
            highs = slice_[slice_ > high_thr]
            if not highs.empty:
                events.append({
                    'channel'     : ch,
                    't_low_cross' : t0,
                    't_high_cross': highs.index[0]
                })

    ev = pd.DataFrame(events).drop_duplicates(subset=['channel','t_low_cross'])
    if ev.empty:
        return ev

    # Compute fluid used and duration
    ev['level_before_L'] = ev['t_low_cross'].map(lambda ts: df[level_col].asof(ts))
    ev['level_after_L']  = ev['t_high_cross'].map(lambda ts: df[level_col].asof(ts))
    ev['fluid_used_L']   = ev['level_before_L'] - ev['level_after_L']
    ev = ev[ev['fluid_used_L'] >= min_fluid].copy()

    ev = ev.sort_values('t_low_cross').reset_index(drop=True)
    ev['cumulative_fluid_used_L'] = ev['fluid_used_L'].cumsum()
    ev['charge_duration_mins'] = (
        ev['t_high_cross'] - ev['t_low_cross']
    ).dt.total_seconds() / 60

    return ev
#############################################################################################################################
def apply_umbilical_charge_rate(
    df: pd.DataFrame,
    events: pd.DataFrame,
    vol_col: str = 'umbilical_charge_volume',
    cum_col: str = 'cum_umbilical_charge_volume'
) -> pd.DataFrame:
    """
    Distributes each event's fluid_used_L evenly across its duration
    and assigns it as volume per row (litres), not rate. Handles irregular time intervals.
    Adds both instantaneous and cumulative charge volume columns.
    """
    df = df.copy()
    df[vol_col] = 0.0

    for _, ev in events.iterrows():
        t0 = ev['t_low_cross']
        t1 = ev['t_high_cross']
        vol = ev['fluid_used_L']

        # mask rows within event duration
        mask = (df.index >= t0) & (df.index <= t1)
        duration_rows = df.loc[mask]

        if not duration_rows.empty:
            per_row_vol = vol / len(duration_rows)
            df.loc[mask, vol_col] = per_row_vol

    # Add cumulative volume column
    df[cum_col] = df[vol_col].cumsum()

    return df

################################################################################################################################################################
def process_umbilical_charges(
    df: pd.DataFrame,
    channel_thresholds: dict,
    level_col: str = 'HPU_SPLY_LEV_L',
    window: pd.Timedelta = pd.Timedelta('2H'),
    min_fluid: float = 5.0,
    vol_col: str = 'umbilical_charge_volume',
    cum_col: str = 'cum_umbilical_charge_volume'
) -> pd.DataFrame:
    """
    Detects umbilical charge events and applies fluid volume across durations.
    Returns updated DataFrame with per-step and cumulative volume columns.
    """
    events = detect_umbilical_charge_events(
        df,
        channel_thresholds=channel_thresholds,
        level_col=level_col,
        window=window,
        min_fluid=min_fluid
    )

    df_updated = apply_umbilical_charge_rate(
        df,
        events,
        vol_col=vol_col,
        cum_col=cum_col
    )

    return df_updated

#####################################################################
def label_valve_pcv_operations(
    df: pd.DataFrame,
    *,
    valve_col: str = 'Valve_Operation_Fluid',
    pcv_col: str   = 'PCV_Fluid_Usage',
    threshold: float = 0.1,                # 0.1 L/min threshold
    threshold_unit: str = "L_per_min",     # "L_per_min" or "L_per_hr"
    min_on_minutes: int = 1,               # keep bursts of at least this many minutes
    bridge_gap_min: int = 0,               # bridge False gaps ≤ this (e.g., 1–2)
    out_col: Optional[str] = None,         # if set, returned df will include this boolean column
    plot: bool = False,
    days_to_show: Optional[int] = 90,
    title: Optional[str] = None,
    return_summary: bool = True
) -> Union[pd.Series, Tuple[pd.Series, Dict[str, Any]], Tuple[pd.Series, Dict[str, Any], pd.DataFrame]]:
    """
    Label timestamps as 'operation' if valve_col > threshold OR pcv_col > threshold.
    Threshold is interpreted as a rate (L/min by default). If your threshold is in L/hr,
    set threshold_unit="L_per_hr" and it will be converted to L/min internally.

    Returns:
      - mask: pd.Series[bool] aligned to df.index
      - meta: dict with 'runs' DataFrame and '% on'
      - df_out: (only if out_col is not None) df with the boolean column added safely
    """
    # ---- checks & prep ----
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex.")
    for c in (valve_col, pcv_col):
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in DataFrame.")

    dat = df[[valve_col, pcv_col]].copy()

    # duplicate-safe: take max across duplicate timestamps
    if dat.index.has_duplicates:
        dat = dat.groupby(level=0).max().sort_index()

    # numeric
    dat[valve_col] = pd.to_numeric(dat[valve_col], errors='coerce')
    dat[pcv_col]   = pd.to_numeric(dat[pcv_col], errors='coerce')

    # units
    unit = threshold_unit.lower().replace(" ", "")
    if unit in ("l_per_hr", "lph", "l/h", "lperhr"):
        thr_lpm = float(threshold) / 60.0
    elif unit in ("l_per_min", "lpm", "l/min", "lpermin"):
        thr_lpm = float(threshold)
    else:
        raise ValueError("threshold_unit must be 'L_per_min' or 'L_per_hr'.")

    # ---- instantaneous operation mask ----
    inst = (dat[valve_col] > thr_lpm) | (dat[pcv_col] > thr_lpm)
    inst = inst.fillna(False)

    # ---- tidy episodes: bridge tiny gaps, drop very short bursts ----
    def enforce_min_and_bridge(mask: pd.Series, min_true=1, bridge=0) -> pd.Series:
        m = mask.copy().fillna(False)
        g = (m != m.shift()).cumsum()
        # bridge short False gaps
        if bridge and bridge > 0:
            for _, grp in m.groupby(g):
                if grp.iloc[0] is False and len(grp) <= bridge:
                    m.loc[grp.index] = True
        # drop short True runs
        g2 = (m != m.shift()).cumsum()
        if min_true and min_true > 1:
            for _, grp in m.groupby(g2):
                if grp.iloc[0] is True and len(grp) < min_true:
                    m.loc[grp.index] = False
        return m

    mask = enforce_min_and_bridge(inst, min_true=int(min_on_minutes), bridge=int(bridge_gap_min))

    # ---- summary ----
    def summarize_runs(mask_bool: pd.Series) -> pd.DataFrame:
        m = mask_bool.fillna(False)
        groups = (m != m.shift()).cumsum()
        rows = []
        for _, grp in m.groupby(groups):
            if bool(grp.iloc[0]):
                rows.append((grp.index[0], grp.index[-1], len(grp)))
        if not rows:
            return pd.DataFrame(columns=['start','end','minutes'])
        return pd.DataFrame(rows, columns=['start','end','minutes'])

    runs = summarize_runs(mask)
    pct_on = 100.0 * mask.mean()
    meta = {"runs": runs, "pct_on": pct_on, "n_events": len(runs)}

    # ---- plot (optional) ----
    if plot:
        sig = pd.concat([dat[valve_col], dat[pcv_col]], axis=1).max(axis=1)
        y = sig; st = mask
        if days_to_show is not None:
            end_ts = sig.index.max()
            start_ts = end_ts - pd.Timedelta(days=days_to_show)
            y = sig.loc[start_ts:end_ts]
            st = mask.reindex(y.index).fillna(False)

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(2,1,1)
        ax1.plot(y.index, y.values, linewidth=0.9, label='max(Valve, PCV) [L/min]')
        # shade operation spans
        on = (st != st.shift()).cumsum()
        first = False
        for _, grp in st.groupby(on):
            if grp.iloc[0]:
                ax1.axvspan(grp.index[0], grp.index[-1], alpha=0.15,
                            label=None if first else 'operation', ymin=0, ymax=1)
                first = True
        ax1.axhline(thr_lpm, linestyle='--', label=f'threshold {thr_lpm:.3f} L/min')
        ttl = title or f"Valve/PCV Operation (thr {threshold} {threshold_unit}, min_on {min_on_minutes} min, bridge ≤{bridge_gap_min} min)"
        ax1.set_title(ttl)
        ax1.set_ylabel('L/min'); ax1.grid(True); ax1.legend(loc='upper left')

        ax2 = fig.add_subplot(2,1,2, sharex=ax1)
        ax2.step(st.index, st.astype(int).values, where='post')
        ax2.set_ylim(-0.1, 1.1); ax2.set_ylabel('operation'); ax2.set_xlabel('Time'); ax2.grid(True)
        plt.tight_layout(); plt.show()

    if out_col is None:
        return (mask, meta) if return_summary else mask

    df_out = df.assign(**{out_col: mask.reindex(df.index).fillna(False)})
    return (mask, meta, df_out) if return_summary else (mask, df_out)

###############################################################################################
def label_fcv_operation_from_col(
    df: pd.DataFrame,
    *,
    col: str = 'FCV_Fluid_Usage',
    threshold: float = 0.3,
    threshold_unit: str = "L_per_min",   # or "L_per_hr"
    min_on_minutes: int = 1,             # drop bursts shorter than this
    bridge_gap_min: int = 0,             # bridge False gaps ≤ this
    out_col: str = 'is_fcv_ops'
):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex.")

    # numeric series
    s = pd.to_numeric(df[col], errors='coerce')

    # convert threshold to L/min if user passed L/hr
    unit = threshold_unit.lower().replace(" ", "")
    thr_lpm = threshold / 60.0 if unit in ("l_per_hr","lph","l/h","lperhr") else float(threshold)

    # instant True where signal exceeds threshold
    inst = (s > thr_lpm).fillna(False)

    # tidy episodes (bridge tiny gaps, drop short bursts)
    def enforce_min_and_bridge(mask: pd.Series, min_true=1, bridge=0) -> pd.Series:
        m = mask.fillna(False).copy()
        g = (m != m.shift()).cumsum()
        if bridge and bridge > 0:
            for _, grp in m.groupby(g):
                if grp.iloc[0] is False and len(grp) <= bridge:
                    m.loc[grp.index] = True
        g2 = (m != m.shift()).cumsum()
        if min_true and min_true > 1:
            for _, grp in m.groupby(g2):
                if grp.iloc[0] is True and len(grp) < min_true:
                    m.loc[grp.index] = False
        return m

    op_mask = enforce_min_and_bridge(inst, min_true=int(min_on_minutes), bridge=int(bridge_gap_min))

    # summary (optional to use)
    groups = (op_mask != op_mask.shift()).cumsum()
    runs = []
    for _, grp in op_mask.groupby(groups):
        if bool(grp.iloc[0]):
            runs.append((grp.index[0], grp.index[-1], len(grp)))
    runs_df = pd.DataFrame(runs, columns=['start','end','minutes']) if runs else pd.DataFrame(columns=['start','end','minutes'])
    pct_on = 100.0 * op_mask.mean()

    # return a new df to avoid SettingWithCopy warnings
    df_out = df.assign(**{out_col: op_mask})
    meta = {"runs": runs_df, "pct_on": pct_on, "n_events": len(runs_df)}
    return op_mask, meta, df_out

#########################################################################################################

def label_tank_fills(
    df: pd.DataFrame,
    *,
    main_period: int = 120,
    main_threshold: float = 30.0,
    backup_period: int = 90,
    backup_threshold: float = 25.0,
    level_col: str = 'HPU_SPLY_LEV_L',
    out_col: str = 'is_tank_fill',
    # episode tidying (optional)
    min_on_minutes: int = 1,     # keep fills at least this long
    bridge_gap_min: int = 0,     # bridge False gaps ≤ this many minutes
    return_summary: bool = False
):
    """
    Label the dataframe with a per-minute boolean column `out_col` indicating tank-fill activity.
    Uses your two-diff method (main + backup) to detect positive level jumps, then optionally
    tidies the mask by bridging tiny gaps and dropping very short bursts.
    """
    d = df.copy()

    # Ensure datetime index & sort
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors='coerce')
    d = d.sort_index()

    # Convert level column to numeric
    s = pd.to_numeric(d[level_col], errors='coerce')

    # Raw fill indicators (your method)
    diff_main   = s.diff(periods=main_period)
    diff_backup = s.diff(periods=backup_period)
    fill_main   = diff_main.where(diff_main > main_threshold, 0.0)
    fill_backup = diff_backup.where(diff_backup > backup_threshold, 0.0)

    d['combined_fill_events'] = pd.concat([fill_main, fill_backup], axis=1).max(axis=1)

    # Instantaneous mask where a jump is detected
    inst_mask = d['combined_fill_events'] > 0

    # Tidy episodes: bridge tiny gaps, drop short bursts
    def enforce_min_and_bridge(mask: pd.Series, min_true=1, bridge=0) -> pd.Series:
        m = mask.fillna(False).copy()
        # bridge short False gaps
        if bridge and bridge > 0:
            g = (m != m.shift()).cumsum()
            for _, grp in m.groupby(g):
                if grp.iloc[0] is False and len(grp) <= bridge:
                    m.loc[grp.index] = True
        # drop short True runs
        if min_true and min_true > 1:
            g2 = (m != m.shift()).cumsum()
            for _, grp in m.groupby(g2):
                if grp.iloc[0] is True and len(grp) < min_true:
                    m.loc[grp.index] = False
        return m

    mask = enforce_min_and_bridge(inst_mask, min_true=int(min_on_minutes), bridge=int(bridge_gap_min))

    # Optional summary of runs
    meta = None
    if return_summary:
        runs = []
        groups = (mask != mask.shift()).cumsum()
        for _, grp in mask.groupby(groups):
            if bool(grp.iloc[0]):
                runs.append((grp.index[0], grp.index[-1], len(grp)))
        runs_df = pd.DataFrame(runs, columns=['start','end','minutes']) if runs else pd.DataFrame(columns=['start','end','minutes'])
        meta = {
            "n_events": len(runs_df),
            "pct_time_fill": 100.0 * mask.mean(),
            "runs": runs_df
        }

    # Return a new frame to avoid SettingWithCopy warnings
    d_out = d.assign(**{out_col: mask})

    return (d_out, meta) if return_summary else d_out
#######################################################################################

# --- 1) Convert events into a boolean label on the dataframe -----------------
def label_is_pressurising_from_events(
    df,
    events,
    out_col='is_pressurising',
    min_on_minutes=1,    # drop very short bursts (< this many rows) if desired
    bridge_gap_min=0     # bridge False gaps ≤ this many rows inside an event block
):
    """
    Given a DataFrame with DatetimeIndex and an events DataFrame that includes
    ['t_low_cross','t_high_cross'], create a boolean column marking rows that
    fall within any event interval [t_low_cross, t_high_cross].

    Returns a NEW DataFrame with 'out_col' attached and a small meta dict.
    """
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors='coerce')
    d = d.sort_index()

    # Empty events -> just False column
    if events is None or events.empty:
        d[out_col] = False
        return d, {"n_events": 0, "pct_time_pressurising": 0.0, "runs": pd.DataFrame(columns=['start','end','rows'])}

    # Build mask: True for any index between t0 and t1 inclusive
    mask = pd.Series(False, index=d.index)
    for _, ev in events.iterrows():
        t0 = ev['t_low_cross']
        t1 = ev['t_high_cross']
        if pd.isna(t0) or pd.isna(t1) or t1 < t0:
            continue
        mask |= (d.index >= t0) & (d.index <= t1)

    # Optional: tidy tiny gaps and short bursts (row-count based)
    def enforce_min_and_bridge(m, min_true=1, bridge=0):
        m = m.fillna(False).copy()
        # bridge short False gaps
        if bridge and bridge > 0:
            g = (m != m.shift()).cumsum()
            for _, grp in m.groupby(g):
                if grp.iloc[0] is False and len(grp) <= bridge:
                    m.loc[grp.index] = True
        # drop short True runs
        if min_true and min_true > 1:
            g2 = (m != m.shift()).cumsum()
            for _, grp in m.groupby(g2):
                if grp.iloc[0] is True and len(grp) < min_true:
                    m.loc[grp.index] = False
        return m

    mask = enforce_min_and_bridge(mask, min_true=int(min_on_minutes), bridge=int(bridge_gap_min))

    # Attach and summarize
    d[out_col] = mask

    runs = []
    groups = (mask != mask.shift()).cumsum()
    for _, grp in mask.groupby(groups):
        if bool(grp.iloc[0]):
            start, end = grp.index[0], grp.index[-1]
            runs.append((start, end, len(grp)))
    runs_df = pd.DataFrame(runs, columns=['start','end','rows']) if runs else pd.DataFrame(columns=['start','end','rows'])

    meta = {
        "n_events": len(runs_df),
        "pct_time_pressurising": 100.0 * mask.mean(),
        "runs": runs_df
    }
    return d, meta


# --- 2) One-call wrapper: detect + (optional) volumes + label ----------------
def detect_and_label_pressurising(
    df,
    channel_thresholds,
    level_col='HPU_SPLY_LEV_L',
    window=pd.Timedelta('2H'),
    min_fluid=5.0,
    out_col='is_pressurising',
    add_volumes=True,                 # if True, calls apply_umbilical_charge_rate
    vol_col='umbilical_charge_volume',
    cum_col='cum_umbilical_charge_volume',
    min_on_minutes=1,
    bridge_gap_min=0
):
    """
    Runs your detect_umbilical_charge_events, optionally applies per-row volumes,
    and labels the dataframe with a boolean `out_col` for pressurising intervals.
    Returns (df_out, events, meta).
    """
    # 1) detect events (your function)
    events = detect_umbilical_charge_events(
        df,
        channel_thresholds=channel_thresholds,
        level_col=level_col,
        window=window,
        min_fluid=min_fluid
    )

    # 2) optionally distribute volumes (your function)
    d = df.copy()
    if add_volumes and events is not None and not events.empty:
        d = apply_umbilical_charge_rate(
            d, events, vol_col=vol_col, cum_col=cum_col
        )

    # 3) label is_pressurising from events
    d_labeled, meta = label_is_pressurising_from_events(
        d, events,
        out_col=out_col,
        min_on_minutes=min_on_minutes,
        bridge_gap_min=bridge_gap_min
    )

    return d_labeled, events, meta

#################################################################


def label_high_external_losses(
    df: pd.DataFrame,
    *,
    col: str = 'External_Losses',      # L/min series at 1-min resolution
    threshold_L_per_day: float = 60.0,
    min_true: int = 15,                  # drop high runs shorter than this (rows/mins)
    bridge: int = 1,                    # bridge False gaps ≤ this
    out_col: str = 'is_losses_high',
    plot: bool = False,
    days_to_show: int = 200
):
    """Label minutes where External_Losses (L/min) >= threshold_L_per_day / 1440."""
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors='coerce')
    d = d.sort_index()

    if col not in d.columns:
        raise KeyError(f"'{col}' not found in DataFrame.")

    s_lpm = pd.to_numeric(d[col], errors='coerce')
    thr_lpm = float(threshold_L_per_day) / 1440.0

    inst = (s_lpm >= thr_lpm).fillna(False)

    # tidy: bridge tiny gaps, drop very short bursts
    def _enforce_min_and_bridge(mask, min_true=1, bridge=0):
        m = mask.fillna(False).copy()
        if bridge and bridge > 0:
            g = (m != m.shift()).cumsum()
            for _, grp in m.groupby(g):
                if grp.iloc[0] is False and len(grp) <= bridge:
                    m.loc[grp.index] = True
        if min_true and min_true > 1:
            g2 = (m != m.shift()).cumsum()
            for _, grp in m.groupby(g2):
                if grp.iloc[0] is True and len(grp) < min_true:
                    m.loc[grp.index] = False
        return m

    mask = _enforce_min_and_bridge(inst, int(min_true), int(bridge))
    d[out_col] = mask

    # summarize runs
    runs = []
    groups = (mask != mask.shift()).cumsum()
    for _, grp in mask.groupby(groups):
        if bool(grp.iloc[0]):
            runs.append((grp.index[0], grp.index[-1], len(grp)))
    runs_df = pd.DataFrame(runs, columns=['start','end','minutes']) if runs else pd.DataFrame(columns=['start','end','minutes'])
    meta = {
        "threshold_L_per_day": threshold_L_per_day,
        "threshold_L_per_min": thr_lpm,
        "pct_time_high": 100.0 * mask.mean(),
        "n_events": len(runs_df),
        "runs": runs_df
    }

    # optional plot (convert series to L/day for readability)
    if plot:
        end = d.index.max()
        start = end - pd.Timedelta(days=days_to_show)
        view = d.loc[(d.index >= start) & (d.index <= end)].copy()

        y_lday = pd.to_numeric(view[col], errors='coerce') * 1440.0  # L/min -> L/day
        st = view[out_col].astype(bool)                               # same index, no reindex()

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(y_lday.index, y_lday.values, lw=0.9, label=col)
        ax.axhline(threshold_L_per_day, ls='--', color='red', lw=1.2,
                   label=f'threshold {threshold_L_per_day:.0f} L/day')

        on = (st != st.shift()).cumsum()
        first = False
        for _, grp in st.groupby(on):
            if grp.iloc[0]:
                ax.axvspan(grp.index[0], grp.index[-1], alpha=0.15,
                           label=None if first else out_col)
                first = True

        ax.set_title(f"{col} — high loss labelling (last {days_to_show} days)")
        ax.set_xlabel("Date / Time"); ax.set_ylabel("L/day")
        ax.grid(True); ax.legend(loc='upper left'); plt.tight_layout(); plt.show()

    return mask, meta, d

##################################################################################################


def add_high_external_losses_label(
    df: pd.DataFrame,
    *,
    col: str = "External_Losses",          # L/min at (about) 1-min resolution
    mode: str = "episode",                  # "episode" or "sliding"

    # --- Option A (episode + hysteresis) ---
    # e.g. 200 L/day -> 200/1440 ≈ 0.139 L/min
    enter_lpm: float = 60.0/1440.0,
    exit_lpm:  float = 56.0/1440.0,        # lower than enter for hysteresis
    min_enter_min: int = 5,                # require ≥45 consecutive mins above enter
    min_exit_min:  int = 3,                 # require ≥5 consecutive mins below exit to end

    # --- Option B (sliding mean) ---
    w_min: int = 60,                        # window for rolling mean (minutes)
    thr_lpm: float = 600.0/1440.0,          # threshold on windowed mean
    min_true_min: int = 5,                 # enforce min duration of True runs
    bridge_gap_min: int = 0,                # optionally join short False gaps

    # --- Output ---
    out_col: str = "is_losses_high",
    return_mask_only: bool = False
):
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors="coerce")
    d = d.sort_index()

    if col not in d.columns:
        raise KeyError(f"Column '{col}' not found.")

    s = pd.to_numeric(d[col], errors="coerce")

    # ---- helpers ----
    def _enforce_min_and_bridge(mask: pd.Series, min_true=1, bridge=0) -> pd.Series:
        m = mask.fillna(False).copy()
        # bridge small False gaps
        if bridge and bridge > 0:
            g = (m != m.shift()).cumsum()
            for _, grp in m.groupby(g):
                if (grp.iloc[0] is False) and (len(grp) <= bridge):
                    m.loc[grp.index] = True
        # drop short True runs
        if min_true and min_true > 1:
            g2 = (m != m.shift()).cumsum()
            for _, grp in m.groupby(g2):
                if (grp.iloc[0] is True) and (len(grp) < min_true):
                    m.loc[grp.index] = False
        return m

    # ---- A) episode + hysteresis (for spikes + persistence) ----
    def _label_episode(series: pd.Series) -> pd.Series:
        vals = series.values.astype(float)
        n = len(vals)
        out = np.zeros(n, dtype=bool)
        in_high = False
        above_cnt = 0
        exit_cnt = 0
        for i, v in enumerate(vals):
            if np.isnan(v):
                above_cnt = 0
                if in_high:
                    exit_cnt += 1
                    if exit_cnt >= min_exit_min:
                        in_high = False; exit_cnt = 0
                continue

            if not in_high:
                if v >= enter_lpm:
                    above_cnt += 1
                    if above_cnt >= min_enter_min:
                        in_high = True
                        start = i - min_enter_min + 1
                        out[start:i+1] = True
                        exit_cnt = 0
                else:
                    above_cnt = 0
            else:
                out[i] = True
                if v <= exit_lpm:
                    exit_cnt += 1
                    if exit_cnt >= min_exit_min:
                        in_high = False; exit_cnt = 0
                else:
                    exit_cnt = 0
        return pd.Series(out, index=series.index, dtype=bool)

    # ---- B) sliding mean above threshold (stable “leak day” feel) ----
    def _label_sliding(series: pd.Series) -> pd.Series:
        rmean = series.rolling(window=w_min, min_periods=w_min).mean()
        base = (rmean >= thr_lpm)
        return _enforce_min_and_bridge(base, min_true=min_true_min, bridge=bridge_gap_min)

    if mode.lower() == "episode":
        mask = _label_episode(s)
        meta_mode = {
            "enter_lpm": float(enter_lpm),
            "exit_lpm": float(exit_lpm),
            "min_enter_min": int(min_enter_min),
            "min_exit_min": int(min_exit_min),
        }
    elif mode.lower() == "sliding":
        mask = _label_sliding(s)
        meta_mode = {
            "w_min": int(w_min),
            "thr_lpm": float(thr_lpm),
            "min_true_min": int(min_true_min),
            "bridge_gap_min": int(bridge_gap_min),
        }
    else:
        raise ValueError("mode must be 'episode' or 'sliding'")

    d[out_col] = mask.astype(bool)

    # summary
    groups = (mask != mask.shift()).cumsum()
    n_events = int(sum(bool(grp.iloc[0]) for _, grp in mask.groupby(groups)))
    meta = {
        "col": col,
        "mode": mode,
        "pct_time_high": float(mask.mean() * 100.0),
        "n_events": n_events,
        **meta_mode
    }

    if return_mask_only:
        return mask.astype(bool), meta
    return d, meta



##################################################################################################

def label_high_pressure(df,
                        lp_col='SCM1_LP_CONS', lp_threshold=245,
                        hp_col='SCM1_HP_CONS', hp_threshold=535,
                        out_lp_col='is_lp_high', out_hp_col='is_hp_high',
                        min_true=1, bridge=0):
    """
    Label high-pressure states for LP and HP systems.
    - Marks 1 where pressure >= threshold, else 0.
    - min_true: drop runs shorter than this (rows)
    - bridge: fill False gaps ≤ this length inside True runs
    """
    d = df.copy()
    
    def _enforce_min_and_bridge(mask, min_true=1, bridge=0):
        m = mask.fillna(False).copy()
        if bridge > 0:
            g = (m != m.shift()).cumsum()
            for _, grp in m.groupby(g):
                if grp.iloc[0] is False and len(grp) <= bridge:
                    m.loc[grp.index] = True
        if min_true > 1:
            g2 = (m != m.shift()).cumsum()
            for _, grp in m.groupby(g2):
                if grp.iloc[0] is True and len(grp) < min_true:
                    m.loc[grp.index] = False
        return m
    
    # LP mask
    lp_mask = pd.to_numeric(d[lp_col], errors='coerce') >= lp_threshold
    lp_mask = _enforce_min_and_bridge(lp_mask, min_true, bridge)
    
    # HP mask
    hp_mask = pd.to_numeric(d[hp_col], errors='coerce') >= hp_threshold
    hp_mask = _enforce_min_and_bridge(hp_mask, min_true, bridge)
    
    # Assign to DataFrame
    d[out_lp_col] = lp_mask.astype(int)
    d[out_hp_col] = hp_mask.astype(int)
    
    return d

########################################################################
def label_pressure_low(df, lp_col='SCM1_LP_CONS', hp_col='SCM1_HP_CONS',
                        lp_threshold=170, hp_threshold=480,
                        out_lp='is_lp_low', out_hp='is_hp_low'):

    df = df.copy()
    # Create flags
    df[out_lp] = df[lp_col] < lp_threshold
    df[out_hp] = df[hp_col] < hp_threshold
    return df

#######################################################
def detect_loss_redundancy(
    df,
    cols=['HPU_LPA_OUT', 'HPU_LPB_OUT', 'HPU_HPA_OUT', 'HPU_HPB_OUT'],
    threshold=100.0,
    out_col='is_no_redundancy',
    plot=False,
    days_to_show=200
):
    """
    Flags no redundancy if ANY of the given pressure columns < threshold (bar).
    Adds a boolean column to the DataFrame.
    """

    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors='coerce')
    d = d.sort_index()

    # Ensure all required columns exist
    missing = [c for c in cols if c not in d.columns]
    if missing:
        raise KeyError(f"Missing columns in DataFrame: {missing}")

    # Flag where any column < threshold
    mask = (d[cols] < threshold).any(axis=1)

    # Add to DataFrame
    d[out_col] = mask

    # Optional plot
    if plot:
        import matplotlib.pyplot as plt
        end_ts = d.index.max()
        start_ts = end_ts - pd.Timedelta(days=days_to_show)
        df_recent = d.loc[start_ts:end_ts, cols + [out_col]]

        fig, ax = plt.subplots(figsize=(14, 6))
        for c in cols:
            ax.plot(df_recent.index, df_recent[c], lw=0.8, label=c)
        ax.axhline(threshold, color='red', linestyle='--', lw=1.2, label=f'Threshold {threshold} bar')

        # Shade no redundancy spans
        mask_recent = df_recent[out_col]
        on = (mask_recent != mask_recent.shift()).cumsum()
        first = False
        for _, grp in mask_recent.groupby(on):
            if grp.iloc[0]:
                ax.axvspan(grp.index[0], grp.index[-1], alpha=0.15,
                           color='red', label=None if first else out_col)
                first = True

        ax.set_title(f"No Redundancy Detection (last {days_to_show} days)")
        ax.set_ylabel("Pressure (bar)")
        ax.grid(True)
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

    return mask, d
###############################################################
def detect_low_supply_tank_level(
    df,
    col='HPU_SPLY_LEV_L',
    threshold=250.0,               # litres
    out_col='is_supply_tank_low',
    plot=False,
    days_to_show=200
):
    """
    Flags when supply tank level (litres) is below threshold.
    Adds a boolean column out_col to the DataFrame.
    """

    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors='coerce')
    d = d.sort_index()

    # Ensure the column exists
    if col not in d.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")

    # Create mask for low tank level
    mask = (d[col] < threshold)

    # Assign to DataFrame
    d[out_col] = mask

    # Optional plot
    if plot:
        import matplotlib.pyplot as plt
        end_ts = d.index.max()
        start_ts = end_ts - pd.Timedelta(days=days_to_show)
        df_recent = d.loc[start_ts:end_ts, [col, out_col]]

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df_recent.index, df_recent[col], lw=0.8, label=col)
        ax.axhline(threshold, color='red', linestyle='--', lw=1.2,
                   label=f'Threshold {threshold} L')

        # Shade low tank spans
        mask_recent = df_recent[out_col]
        on = (mask_recent != mask_recent.shift()).cumsum()
        first = False
        for _, grp in mask_recent.groupby(on):
            if grp.iloc[0]:
                ax.axvspan(grp.index[0], grp.index[-1], alpha=0.15,
                           color='red', label=None if first else out_col)
                first = True

        ax.set_title(f"Low Supply Tank Level Detection (last {days_to_show} days)")
        ax.set_ylabel("Litres")
        ax.grid(True)
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

    return mask, d

###################################################################
import numpy as np
import pandas as pd

def add_pump_states(
    df,
    hp_cum_col="Cum_HP_Pump_A_Run_Count",
    lp_cum_col="Cum_LP_Pump_A_Run_Count",
    lookback_days=120,
    q_elev=0.70, q_high=0.95,
    floors_hp=(1.2, 1.6),      # HP floors (elev, high) in starts/h
    floors_lp=(0.25, 0.75),    # LP floors
    gap_min=0.25,
    # separate logic + persistence per level
    logic_elev="or",           # 'or' catches short bursts
    logic_high="and",          # 'and' demands sustained demand
    persist_elev_min=10,       # minutes required to ENTER elevated
    persist_high_min=30,       # minutes required to ENTER high
    exit_ratio=0.90,           # hysteresis (leave when below exit_ratio*thresholds)
    exclude_fill_col="is_tank_fill",  # optionally suppress flags during refills
    inplace=True,
):
    """
    Adds minute-level pump state & rate columns with the names:
      HP_pump_rate_2h,  HP_pump_rate_24h,  HP_pump_state,  is_HP_pump_normal
      LP_pump_rate_2h,  LP_pump_rate_24h,  LP_pump_state,  is_LP_pump_normal

    State codes: 1=steady (includes 'off'), 2=elevated, 3=high.

    Robust to duplicate timestamps: cumulative counters are resampled to an exact
    1-minute grid using last-observation-carried-forward before computing rates.
    """

    # ---- helpers ----
    def _prep(d):
        out = d if inplace else d.copy()
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out

    def _resample_min(s):
        i0, i1 = s.index.min().floor("T"), s.index.max().ceil("T")
        idx = pd.date_range(i0, i1, freq="T")
        return s.resample("T").last().reindex(idx).ffill()

    def _th(series, floors):
        base = series.loc[series.index >= series.index.max() - pd.Timedelta(days=lookback_days)]
        if not base.notna().any():
            base = series
        Te = float(np.nanquantile(base, q_elev)) if base.notna().any() else 0.0
        Th = float(np.nanquantile(base, q_high)) if base.notna().any() else 0.0
        Te, Th = max(Te, floors[0]), max(Th, floors[1])
        if Th <= Te:
            Th = Te + gap_min
        return Te, Th

    def _persist(mask, mins):
        if mins <= 1:
            return mask.astype(bool)
        x = mask.astype("int8").to_numpy()
        out = np.zeros_like(x, dtype=bool)
        run = 0
        for i, v in enumerate(x):
            run = run + 1 if v else 0
            out[i] = run >= mins
        return pd.Series(out, index=mask.index)

    def _classify(cum_min, floors, base_prefix):
        """
        base_prefix: 'HP' or 'LP' (used for column names)
        Emits: <base>_pump_rate_2h, <base>_pump_rate_24h, <base>_pump_state, is_<base>_pump_normal
        """
        # per-minute starts from cumulative counter
        starts = cum_min.diff().clip(lower=0).fillna(0.0)

        # rolling sums -> starts/hour
        r2  = (starts.rolling(120,  min_periods=1).sum() / 2.0).astype(float)
        r24 = (starts.rolling(1440, min_periods=1).sum() / 24.0).astype(float)

        # thresholds (per window)
        T2e, T2h   = _th(r2,  floors)
        T24e, T24h = _th(r24, floors)

        # enter conditions
        elev = (r2 >= T2e) | (r24 >= T24e) if logic_elev == "or" else (r2 >= T2e) & (r24 >= T24e)
        high = (r2 >= T2h) | (r24 >= T24h) if logic_high == "or" else (r2 >= T2h) & (r24 >= T24h)

        # persistence to enter
        elev = _persist(elev, persist_elev_min)
        high = _persist(high, persist_high_min)

        # hysteresis to exit
        T2e_x, T2h_x   = T2e * exit_ratio,  T2h * exit_ratio
        T24e_x, T24h_x = T24e * exit_ratio, T24h * exit_ratio

        idx = r2.index
        a, b = r2.to_numpy(), r24.to_numpy()
        e_en, h_en = elev.to_numpy(), high.to_numpy()

        state = np.empty(len(idx), dtype=np.int8)
        cur = 1  # 1=steady/off
        for i in range(len(idx)):
            if cur != 3 and h_en[i]:
                cur = 3
            elif cur == 3 and not ((a[i] >= T2h_x) or (b[i] >= T24h_x)):
                cur = 2 if ((a[i] >= T2e) or (b[i] >= T24e)) else 1
            elif cur == 1 and e_en[i]:
                cur = 2
            elif cur == 2 and not ((a[i] >= T2e_x) or (b[i] >= T24e_x)):
                cur = 1
            state[i] = cur

        # build output with requested names
        rate2_col   = f"{base_prefix}_pump_rate_2h"
        rate24_col  = f"{base_prefix}_pump_rate_24h"
        state_col   = f"{base_prefix}_pump_state"
        normal_col  = f"is_{base_prefix}_pump_normal"

        out = pd.DataFrame({
            rate2_col:  r2,
            rate24_col: r24,
            state_col:  pd.Series(state, index=idx),
        }, index=idx)

        # optionally suppress during tank fills
        if exclude_fill_col and exclude_fill_col in df.columns:
            fill = df.resample("T")[exclude_fill_col].max().reindex(idx).fillna(False).astype(bool)
            out.loc[fill, state_col] = 1

        out[normal_col] = (out[state_col] == 1)

        thr = {"2h_elev": T2e, "2h_high": T2h, "24h_elev": T24e, "24h_high": T24h}
        return out, thr

    # ---- run on both pumps ----
    df = _prep(df)

    if hp_cum_col not in df.columns or lp_cum_col not in df.columns:
        missing = [c for c in (hp_cum_col, lp_cum_col) if c not in df.columns]
        raise KeyError(f"Missing required columns: {missing}")

    # resample cumulative counters to exact 1-minute grid
    hp_cum_min = _resample_min(df[hp_cum_col].astype(float))
    lp_cum_min = _resample_min(df[lp_cum_col].astype(float))

    hp_out, thr_hp = _classify(hp_cum_min, floors_hp, "HP")
    lp_out, thr_lp = _classify(lp_cum_min, floors_lp, "LP")

    # overwrite any previous outputs cleanly, then join
    df.drop(columns=list(hp_out.columns) + list(lp_out.columns), errors="ignore", inplace=True)
    df = df.join(hp_out, how="left").join(lp_out, how="left")

    return (df if inplace else df.copy()), {"HP": thr_hp, "LP": thr_lp}
###############################################################################
########################################################################################
def add_fcv_operation_simple(
    df: pd.DataFrame,
    col: str = "FCV_Fluid_Usage",
    threshold: float = 0.3,
    out_col: str = "FCV_Operation",
) -> tuple[pd.DataFrame, dict]:
    """
    Adds a simple boolean FCV operation column:
      True  -> value > threshold
      False -> value <= threshold or NaN/invalid
    Returns (df, meta).
    """
    s = pd.to_numeric(df[col], errors="coerce").gt(threshold).fillna(False)
    df = df.copy()
    df[out_col] = s.astype(bool)
    meta = {
        "source_col": col,
        "threshold_L_per_min": threshold,
        "true_count": int(s.sum()),
        "true_pct": float(s.mean() * 100.0),
    }
    return df, meta
##################################################################################################################################################################



def run_labelling_pipeline(
    df: pd.DataFrame,
    *,
    channel_thresholds: dict,
    pressurising_level_col: str = 'HPU_SPLY_LEV_L',
    plot: bool = False
) -> tuple[pd.DataFrame, dict]:
    """
    Applies all labeling functions in a consistent order and returns
    (labeled_df, metas). Plotting is disabled by default for speed.
    """
    metas: dict = {}

    # 1) Low fluid consumption (adds boolean column)
    df = add_low_fluid_consumption_label(
        df,
        col="Supply_Consumption_Excl_Fills",
        mode="episode",          # or "sliding"
        enter_lpm=63/24/60, exit_lpm=65/24/60,
        min_enter_min=2, min_exit_min=2,
        out_col="is_consumption_low",
        return_mask_only=False
    )

    # 2) Valve/PCV operation mask
    op_mask, op_meta, df = label_valve_pcv_operations(
        df,
        valve_col='Valve_Operation_Fluid',
        pcv_col='PCV_Fluid_Usage',
        threshold=0.1, threshold_unit="L_per_min",
        min_on_minutes=1, bridge_gap_min=1,
        out_col='is_valve_ops',
        plot=plot
    )
    metas['valve_pcv_ops'] = op_meta

    # 3) FCV operation mask (single column)
    fcv_mask, fcv_meta, df = label_fcv_operation_from_col(
        df,
        col='FCV_Fluid_Usage',
        threshold=0.3, threshold_unit="L_per_min",
        min_on_minutes=1, bridge_gap_min=1,
        out_col='is_fcv_ops'
    )
    metas['fcv_ops'] = fcv_meta

    # 3a) Simple per-sample FCV flag for quick diagnostics / QC
    df, fcv_simple_meta = add_fcv_operation_simple(
        df,
        col='FCV_Fluid_Usage',
        threshold=0.3,
        out_col='FCV_Operation'
    )
    metas['fcv_ops_simple'] = fcv_simple_meta

    # 4) Tank fills (returns df + meta)
    df, fills_meta = label_tank_fills(
        df,
        level_col='HPU_SPLY_LEV_L',
        out_col='is_tank_fill',
        main_period=120, main_threshold=30.0,
        backup_period=90, backup_threshold=25.0,
        min_on_minutes=3, bridge_gap_min=1,
        return_summary=True
    )
    metas['tank_fills'] = fills_meta

    # 5) Pressurising (detect events -> optional volumes -> label)
    df, charge_events, press_meta = detect_and_label_pressurising(
        df,
        channel_thresholds=channel_thresholds,
        level_col=pressurising_level_col,
        window=pd.Timedelta('2H'),
        min_fluid=5.0,
        out_col='is_pressurising',
        add_volumes=True,
        vol_col='umbilical_charge_volume',
        cum_col='cum_umbilical_charge_volume',
        min_on_minutes=2, bridge_gap_min=1
    )
    metas['pressurising'] = press_meta
    metas['pressurising_events'] = charge_events

    # 6) High external losses
   # _, loss_meta, df = label_high_external_losses(
   #     df,
   #     col='External_Losses',
   #     threshold_L_per_day=1157.5,
   #     min_true=15, bridge=0,
   #     plot=False,
   #     days_to_show=30,
   #     out_col='is_losses_high'
   # )
   # metas['losses_high'] = loss_meta
   
    df, loss_meta = add_high_external_losses_label(
        df,
        col="External_Losses",
        mode="episode",          # <- switch mode
        enter_lpm=92.5/1440,      # trigger when ≥200 L/day (in L/min)
        exit_lpm=91.5/1440,       # exit when ≤150 L/day (in L/min)
        min_enter_min=2,        # need ≥45 consecutive mins to confirm
        min_exit_min=1,          # need ≥5 consecutive mins to clear
        out_col="is_losses_high",
    )
    metas["losses_high"] = loss_meta

    # 7) High/low pressure flags
    df = label_high_pressure(
        df,
        lp_col='SCM1_LP_CONS', lp_threshold=245,
        hp_col='SCM1_HP_CONS', hp_threshold=537,
        out_lp_col='is_lp_high', out_hp_col='is_hp_high',
        min_true=2, bridge=1
    )
    df = label_pressure_low(
        df,
        lp_col='SCM1_LP_CONS', hp_col='SCM1_HP_CONS',
        lp_threshold=170, hp_threshold=480,
        out_lp='is_lp_low', out_hp='is_hp_low'
    )

    # 8) Redundancy and tank level (return mask + df)
    mask_nr, df = detect_loss_redundancy(
        df,
        threshold=100.0,
        plot=False
    )
    df['is_no_redundancy'] = mask_nr

    mask_low_tank, df = detect_low_supply_tank_level(
        df,
        threshold=250.0,
        plot=False
    )
    df['is_supply_tank_low'] = mask_low_tank

    # 9) Pump states (adds several columns, returns thresholds)
    df, pump_thr = add_pump_states(
        df,
        logic_elev="or", logic_high="or",
        persist_elev_min=10, persist_high_min=30,
        exit_ratio=0.90,
        exclude_fill_col="is_tank_fill",
        floors_hp=(1.2, 1.6),
        floors_lp=(0.25, 0.75),
    )
    metas['pump_thresholds'] = pump_thr

    return df, metas

#do somehting with the delta pressures 

