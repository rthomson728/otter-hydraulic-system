from __future__ import annotations

# Standard libs
from pathlib import Path
import os
import re
from typing import Optional, Tuple, Dict, Any, Union, List

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# (These are imported elsewhere in your file if needed)
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# import shap
# from IPython.display import HTML, display
# import matplotlib.dates as mdates
# from scipy.stats import pearsonr, spearmanr
# from scipy.fft import fft, fftfreq
# from sklearn.linear_model import LinearRegression

# =============================================================================
# Helpers
# =============================================================================

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a DatetimeIndex (coerced, dropped NaT, sorted)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        d = df.copy()
        d.index = pd.to_datetime(d.index, errors="coerce")
    else:
        d = df.copy()
    d = d[~d.index.isna()].sort_index()
    return d

def _subset_by_date(df: pd.DataFrame,
                    start_date: Optional[Union[str, pd.Timestamp]] = None,
                    end_date: Optional[Union[str, pd.Timestamp]] = None) -> pd.DataFrame:
    """Slice by optional start/end (inclusive), tolerant of None."""
    if start_date is not None:
        df = df.loc[pd.to_datetime(start_date):]
    if end_date is not None:
        # include the end_date instant (assume timestamps; if daily, adjust as needed)
        df = df.loc[:pd.to_datetime(end_date)]
    return df

# =============================================================================
# Plots
# =============================================================================

def plot_hpu_data(df: pd.DataFrame,
                  supply_col: str,
                  lp_pressure_col: str,
                  hp_pressure_col: str,
                  start_date: Optional[Union[str, pd.Timestamp]] = None,
                  end_date: Optional[Union[str, pd.Timestamp]] = None
                 ) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot HPU Supply Tank Level vs HP & LP Pressure over an optional date window.
    """
    d = _ensure_dt_index(df)
    cols = [supply_col, lp_pressure_col, hp_pressure_col]
    missing = [c for c in cols if c not in d.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    d = d[cols]
    d = _subset_by_date(d, start_date, end_date)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(d.index, d[supply_col], label='Supply Tank Level %', linewidth=1.2)
    ax.plot(d.index, d[lp_pressure_col], label='LP Pressure (A)', linewidth=1.2)
    ax.plot(d.index, d[hp_pressure_col], label='HP Pressure (A)', linewidth=1.2)

    ax.set_title('HPU Supply Tank vs HP & LP Pressure')
    ax.set_xlabel('Date')
    ax.set_ylabel('Pressure / Level')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()
    return fig, [ax]

def plot_low_consumption(df: pd.DataFrame,
                         value_col: str = "Supply_Consumption_Excl_Fills",
                         state_col: str = "is_consumption_low",
                         days: int = 30,
                         threshold_lpm: Optional[float] = None,
                         resample: Optional[str] = "5T"
                        ) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Visualize low-consumption state shading over the chosen window.
    """
    d = _ensure_dt_index(df)

    if value_col not in d.columns or state_col not in d.columns:
        missing = [c for c in [value_col, state_col] if c not in d.columns]
        raise KeyError(f"Missing columns: {missing}")

    end = d.index.max()
    start = end - pd.Timedelta(days=days)

    y = pd.to_numeric(d[value_col], errors="coerce").loc[start:end]
    st = d[state_col].astype(bool).reindex(y.index, fill_value=False)

    if resample:
        y_plot = y.resample(resample).median()
        st_plot = st.resample(resample).max().astype(bool)
    else:
        y_plot, st_plot = y, st

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_plot.index, y_plot.values, lw=0.9, label=value_col)
    if threshold_lpm is not None:
        ax.axhline(threshold_lpm, ls="--", lw=1.2, label=f"threshold {threshold_lpm:.3f} L/min")

    # Shade low-consumption spans
    runs = (st_plot != st_plot.shift()).cumsum()
    first = True
    for _, grp in st_plot.groupby(runs):
        if grp.iloc[0]:
            ax.axvspan(grp.index[0], grp.index[-1], alpha=0.15,
                       label="low consumption" if first else None)
            first = False

    ax.set_title(f"{value_col} — low consumption (last {days} days)")
    ax.set_ylabel("L/min")
    ax.grid(True)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    return fig, [ax]

def plot_losses_daily(labeled_df: pd.DataFrame,
                      clip_upper_day: float = 500
                     ) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Highlight whole days where any time within that day was flagged high.
    Requires:
      - 'External_Losses' (litres)
      - 'is_losses_high' (boolean)
    """
    d = _ensure_dt_index(labeled_df)
    for col in ["External_Losses", "is_losses_high"]:
        if col not in d.columns:
            raise KeyError(f"Missing column: {col}")

    daily_losses = d["External_Losses"].resample("D").sum()
    daily_labels = d["is_losses_high"].resample("D").max()

    end = daily_losses.index.max()
    start = end - pd.Timedelta(days=30)
    daily_losses = daily_losses.loc[start:end].clip(upper=clip_upper_day)
    daily_labels = daily_labels.loc[start:end]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_losses.index, daily_losses.values,
            label=f"External losses (clipped ≤{clip_upper_day} L/day)")
    ax.plot(daily_losses.index,
            daily_losses.rolling(7, min_periods=1).mean(),
            linewidth=2, label="7-day rolling mean (daily)")

    for date, high in daily_labels.items():
        if bool(high):
            ax.axvspan(date, date + pd.Timedelta(days=1), alpha=0.2)

    ax.set_title("External Losses – last 30 days (high-loss DAYS highlighted)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Litres per day")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    return fig, [ax]

def plot_losses_hourly(labeled_df: pd.DataFrame,
                       clip_upper_hour: float = 100,
                       rolling_days: int = 7
                      ) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Highlight only the actual HOURS that are flagged high.
    Requires:
      - 'External_Losses' (litres)
      - 'is_losses_high' (boolean)
    """
    d = _ensure_dt_index(labeled_df)
    for col in ["External_Losses", "is_losses_high"]:
        if col not in d.columns:
            raise KeyError(f"Missing column: {col}")

    hourly_losses = d["External_Losses"].resample("H").sum()
    hourly_labels = d["is_losses_high"].resample("H").max()

    end = hourly_losses.index.max()
    start = end - pd.Timedelta(days=30)
    hourly_losses = hourly_losses.loc[start:end].clip(upper=clip_upper_hour)
    hourly_labels = hourly_labels.loc[start:end]

    win = max(1, rolling_days * 24)  # guard

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hourly_losses.index, hourly_losses.values,
            label=f"External losses (clipped ≤{clip_upper_hour} L/hour)")
    ax.plot(hourly_losses.index,
            hourly_losses.rolling(win, min_periods=1).mean(),
            linewidth=2, label=f"{rolling_days}-day rolling mean (hourly)")

    for date, high in hourly_labels.items():
        if bool(high):
            ax.axvspan(date, date + pd.Timedelta(hours=1), alpha=0.2)

    ax.set_title("External Losses – last 30 days (high-loss HOURS highlighted)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Litres per hour")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    return fig, [ax]

def plot_external_losses_high(df: pd.DataFrame,
                              value_col: str = "External_Losses",
                              state_col: str = "is_losses_high",
                              days: int = 30,
                              threshold_lpm: Optional[float] = None,  # e.g. 90/1440
                              resample: Optional[str] = "5T",          # None to keep native
                              clip_upper: Optional[float] = None
                             ) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Episode-style highlight of high losses over the selected window.
    """
    d = _ensure_dt_index(df)
    for col in [value_col, state_col]:
        if col not in d.columns:
            raise KeyError(f"Missing column: {col}")

    end = d.index.max()
    start = end - pd.Timedelta(days=days)

    y = pd.to_numeric(d[value_col], errors="coerce").loc[start:end]
    st = d[state_col].astype(bool).reindex(y.index, fill_value=False)

    if resample:
        y_plot = y.resample(resample).median()
        st_plot = st.resample(resample).max().astype(bool)
    else:
        y_plot, st_plot = y, st

    if clip_upper is not None:
        y_plot = y_plot.clip(upper=clip_upper)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_plot.index, y_plot.values, lw=0.9, label=value_col)

    if threshold_lpm is not None:
        ax.axhline(threshold_lpm, ls="--", lw=1.2,
                   label=f"threshold {threshold_lpm:.5f} L/min")

    runs = (st_plot != st_plot.shift()).cumsum()
    first = True
    for _, grp in st_plot.groupby(runs):
        if grp.iloc[0]:
            ax.axvspan(grp.index[0], grp.index[-1], alpha=0.15,
                       label="high-loss episode" if first else None)
            first = False

    ax.set_title(f"{value_col} — high-loss episodes (last {days} days)")
    ax.set_ylabel("L/min")
    ax.grid(True)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    return fig, [ax]

def plot_valve_pcv_ops(df: pd.DataFrame,
                       valve_col: str = "Valve_Operation_Fluid",
                       pcv_col: str = "PCV_Fluid_Usage",
                       state_col: str = "is_valve_ops",
                       days: int = 90,
                       threshold_lpm: Optional[float] = None
                      ) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Visualize combined valve/PCV activity and the operation state strip.
    """
    d = _ensure_dt_index(df)
    for col in [valve_col, pcv_col, state_col]:
        if col not in d.columns:
            raise KeyError(f"Missing column: {col}")

    sig = pd.concat([pd.to_numeric(d[valve_col], errors='coerce'),
                     pd.to_numeric(d[pcv_col],   errors='coerce')], axis=1).max(axis=1)

    end = sig.index.max()
    start = end - pd.Timedelta(days=days)
    y = sig.loc[start:end]
    st = d[state_col].astype(bool).reindex(y.index, fill_value=False)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(y.index, y.values, lw=0.9, label='max(Valve, PCV) [L/min]')

    runs = (st != st.shift()).cumsum()
    first = True
    for _, grp in st.groupby(runs):
        if grp.iloc[0]:
            ax1.axvspan(grp.index[0], grp.index[-1], alpha=0.15,
                        label='operation' if first else None)
            first = False

    if threshold_lpm is not None:
        ax1.axhline(threshold_lpm, ls='--', lw=1.2,
                    label=f'thr {threshold_lpm:.3f} L/min')

    ax1.set_title(f"Valve/PCV Operation (last {days} days)")
    ax1.set_ylabel('L/min')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    ax2.step(st.index, st.astype(int).values, where='post')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_ylabel('operation')
    ax2.set_xlabel('Time')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    return fig, [ax1, ax2]

def plot_fcv_usage_vs_operation_line(df: pd.DataFrame,
                                     usage_col: str = "FCV_Fluid_Usage",   # L/min
                                     op_col: str = "FCV_Operation",        # boolean or 0/1
                                     days_to_show: int = 90,
                                     resample_rule: Optional[str] = "5T",  # None to keep native
                                     rolling_minutes: Optional[int] = None,
                                     title: Optional[str] = None
                                    ) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot FCV usage (line) with shaded operation episodes and a state strip.
    """
    d = _ensure_dt_index(df)
    for col in [usage_col]:
        if col not in d.columns:
            raise KeyError(f"Missing column: {col}")
    if op_col not in d.columns:
        # default to all False if not present
        d[op_col] = False

    y_raw = pd.to_numeric(d[usage_col], errors="coerce")
    st_raw = d[op_col].astype(bool)

    end = y_raw.index.max()
    start = end - pd.Timedelta(days=days_to_show)
    y_raw = y_raw.loc[start:end]
    st_raw = st_raw.loc[start:end].reindex(y_raw.index).fillna(False)

    if resample_rule:
        y = y_raw.resample(resample_rule).mean()
        st = st_raw.resample(resample_rule).max().astype(bool)
    else:
        y, st = y_raw, st_raw

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(2, 1, 1)

    ax1.plot(y.index, y.values, linewidth=1.0, alpha=0.9, label=f"{usage_col} (L/min)")

    # Optional rolling mean overlay (guard for <2 points)
    if rolling_minutes and len(y.index) > 1:
        # infer step from index
        step = (y.index[1] - y.index[0])
        if step > pd.Timedelta(0):
            win = max(1, int(pd.Timedelta(minutes=rolling_minutes) / step))
            ax1.plot(y.index, y.rolling(win, min_periods=1).mean(),
                     linewidth=2.0, label=f"{rolling_minutes}-min rolling mean")

    # Shade operation spans
    runs = (st != st.shift()).cumsum()
    first = True
    for _, grp in st.groupby(runs):
        if grp.iloc[0]:
            ax1.axvspan(grp.index[0], grp.index[-1], alpha=0.15,
                        label="FCV_Operation" if first else None)
            first = False

    ax1.set_title(title or f"FCV usage vs operation (last {days_to_show} days)")
    ax1.set_ylabel("L/min")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    ax2.step(st.index, st.astype(int).values, where="post")
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_ylabel(op_col)
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig, [ax1, ax2]
###########################################
def plot_level_vs_fill_label(
    df: pd.DataFrame,
    level_col: str = 'HPU_SPLY_LEV_L',
    label_col: str = 'is_tank_fill',
    days_to_show: int = 90,
    title: Optional[str] = None
):
    """
    Plot the tank level with shaded spans where `label_col` is True,
    plus a binary state strip underneath.
    """
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors='coerce')
    d = d.sort_index()

    if level_col not in d.columns:
        raise KeyError(f"'{level_col}' not found in DataFrame.")
    if label_col not in d.columns:
        raise KeyError(f"'{label_col}' not found. Create it first (e.g., with label_tank_fills).")

    y = pd.to_numeric(d[level_col], errors='coerce')
    st = d[label_col].astype(bool)

    # limit window
    end = y.index.max()
    start = end - pd.Timedelta(days=days_to_show)
    y = y.loc[start:end]
    # duplicate-safe alignment of the label
    try:
        st = st.loc[y.index]
    except KeyError:
        st = st.reindex(y.index)
    st = st.fillna(False)

    # plot
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(y.index, y.values, linewidth=0.9, label=level_col)

    # shade True spans
    on = (st != st.shift()).cumsum()
    first = False
    for _, grp in st.groupby(on):
        if grp.iloc[0]:
            ax1.axvspan(grp.index[0], grp.index[-1], alpha=0.15,
                        label=None if first else label_col, ymin=0, ymax=1)
            first = True

    ax1.set_title(title or f"{level_col} with '{label_col}' (last {days_to_show} days)")
    ax1.set_ylabel('Level (L)')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    # state strip
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax2.step(st.index, st.astype(int).values, where='post')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_ylabel(label_col)
    ax2.set_xlabel('Time')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    ##############################################################
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List

def plot_pressurising_vs_channels(
    df: pd.DataFrame,
    channels: List[str],
    label_col: str = 'is_pressurising',
    days_to_show: int = 90,
    title: Optional[str] = None
):
    """
    Plot multiple channel time series with shaded spans where pressurising label is True.
    Handles duplicate timestamps in index.
    """
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors='coerce')
    d = d.sort_index()

    # Check columns exist
    for ch in channels:
        if ch not in d.columns:
            raise KeyError(f"'{ch}' not found in DataFrame.")
    if label_col not in d.columns:
        raise KeyError(f"'{label_col}' not found in DataFrame.")

    # Limit to last N days
    end = d.index.max()
    start = end - pd.Timedelta(days=days_to_show)
    d = d.loc[(d.index >= start) & (d.index <= end)]

    # Get state mask aligned with current view
    st = pd.Series(d[label_col].values, index=d.index).astype(bool).fillna(False)

    fig, axes = plt.subplots(len(channels)+1, 1, figsize=(14, 2*(len(channels)+1)), sharex=True)

    for i, ch in enumerate(channels):
        ax = axes[i]
        y = pd.to_numeric(d[ch], errors='coerce')
        ax.plot(y.index, y.values, label=ch, linewidth=0.9)

        # Shade pressurising spans
        on = (st != st.shift()).cumsum()
        first = False
        for _, grp in st.groupby(on):
            if grp.iloc[0]:
                ax.axvspan(grp.index[0], grp.index[-1], alpha=0.15,
                           label=None if first else label_col)
                first = True

        ax.set_ylabel(ch)
        ax.grid(True)
        ax.legend(loc='upper left')

    # Binary state strip
    ax_state = axes[-1]
    ax_state.step(st.index, st.astype(int).values, where='post')
    ax_state.set_ylim(-0.1, 1.1)
    ax_state.set_ylabel(label_col)
    ax_state.set_xlabel('Time')
    ax_state.grid(True)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()

####################################################################

def evaluate_pressure_flags(
    df: pd.DataFrame,
    lp_col='SCM1_LP_CONS', hp_col='SCM1_HP_CONS',
    lp_thr=245, hp_thr=537,
    lp_flag_col='is_lp_high', hp_flag_col='is_hp_high',
    days=200, min_true=None, bridge=None,  # not used here (flags already set); kept for symmetry
    make_plots=True
):
    d = df.copy()
    # ensure datetime index
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors='coerce')
    d = d.sort_index()

    # slice last N days
    end = d.index.max()
    start = end - pd.Timedelta(days=days)
    d = d.loc[(d.index >= start) & (d.index <= end)].copy()

    # numeric series
    lp = pd.to_numeric(d[lp_col], errors='coerce')
    hp = pd.to_numeric(d[hp_col], errors='coerce')

    # ground-truth masks from thresholds
    gt_lp = (lp >= lp_thr)
    gt_hp = (hp >= hp_thr)

    # use existing flags if present; else create them from thresholds
    if lp_flag_col in d.columns:
        flag_lp = d[lp_flag_col].astype(bool)
    else:
        flag_lp = gt_lp.copy()
        d[lp_flag_col] = flag_lp.astype(int)

    if hp_flag_col in d.columns:
        flag_hp = d[hp_flag_col].astype(bool)
    else:
        flag_hp = gt_hp.copy()
        d[hp_flag_col] = flag_hp.astype(int)

    # accuracy calculations (ignore rows where tag is NaN)
    def accuracy(gt: pd.Series, flag: pd.Series) -> dict:
        valid = gt.notna()
        n = valid.sum()
        if n == 0:
            return {"n": 0, "acc": np.nan, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
        gt_v = gt[valid].astype(bool)
        fl_v = flag[valid].astype(bool)

        tp = ((gt_v) & (fl_v)).sum()
        tn = ((~gt_v) & (~fl_v)).sum()
        fp = ((~gt_v) & (fl_v)).sum()
        fn = ((gt_v) & (~fl_v)).sum()
        acc = (tp + tn) / n * 100.0
        return {"n": int(n), "acc": acc, "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}

    lp_metrics = accuracy(gt_lp, flag_lp)
    hp_metrics = accuracy(gt_hp, flag_hp)

    print(f"[LP]  threshold ≥ {lp_thr} | minutes evaluated: {lp_metrics['n']:,} | "
          f"accuracy: {lp_metrics['acc']:.2f}%  "
          f"(TP={lp_metrics['tp']}, TN={lp_metrics['tn']}, FP={lp_metrics['fp']}, FN={lp_metrics['fn']})")
    print(f"[HP]  threshold ≥ {hp_thr} | minutes evaluated: {hp_metrics['n']:,} | "
          f"accuracy: {hp_metrics['acc']:.2f}%  "
          f"(TP={hp_metrics['tp']}, TN={hp_metrics['tn']}, FP={hp_metrics['fp']}, FN={hp_metrics['fn']})")

    if make_plots:
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

        # --- LP plot ---
        ax = axes[0]
        ax.plot(lp.index, lp.values, lw=0.9, label=lp_col)
        ax.axhline(lp_thr, ls='--', lw=1.0, label=f'LP threshold {lp_thr}')
        # shade flagged periods
        on = (flag_lp != flag_lp.shift()).cumsum()
        first = False
        for _, grp in flag_lp.groupby(on):
            if grp.iloc[0]:
                ax.axvspan(grp.index[0], grp.index[-1], alpha=0.12,
                           label=None if first else lp_flag_col)
                first = True
        # mark mismatches (flag != ground truth)
        mismatch_lp = (flag_lp != gt_lp) & lp.notna()
        ax.scatter(lp.index[mismatch_lp], lp[mismatch_lp], marker='x', s=12, label='LP mismatch')
        ax.set_ylabel('LP value'); ax.grid(True); ax.legend(loc='upper left')
        ax.set_title(f"LP trend & flag (last {days} days)")

        # --- HP plot ---
        ax = axes[1]
        ax.plot(hp.index, hp.values, lw=0.9, label=hp_col)
        ax.axhline(hp_thr, ls='--', lw=1.0, label=f'HP threshold {hp_thr}')
        on = (flag_hp != flag_hp.shift()).cumsum()
        first = False
        for _, grp in flag_hp.groupby(on):
            if grp.iloc[0]:
                ax.axvspan(grp.index[0], grp.index[-1], alpha=0.12,
                           label=None if first else hp_flag_col)
                first = True
        mismatch_hp = (flag_hp != gt_hp) & hp.notna()
        ax.scatter(hp.index[mismatch_hp], hp[mismatch_hp], marker='x', s=12, label='HP mismatch')
        ax.set_ylabel('HP value'); ax.set_xlabel('Time'); ax.grid(True); ax.legend(loc='upper left')
        ax.set_title(f"HP trend & flag (last {days} days)")

        plt.tight_layout()
        plt.show()

    return d, lp_metrics, hp_metrics
#########################################

def plot_lp_hp_flags(labeled_df, days_to_show=200,
                     lp_col="SCM1_LP_CONS", lp_flag_col="is_lp_low", lp_thresh=170,
                     hp_col="SCM1_HP_CONS", hp_flag_col="is_hp_low", hp_thresh=480):
    """
    Plot LP and HP pressures with shaded low-pressure flag regions.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        DataFrame with datetime index and pressure/flag columns.
    days_to_show : int, default=200
        Number of days from the end of the dataset to display.
    lp_col, lp_flag_col, lp_thresh : str, str, float
        Column names and threshold for LP pressure and its flag.
    hp_col, hp_flag_col, hp_thresh : str, str, float
        Column names and threshold for HP pressure and its flag.
    """

    # Define window
    end_ts = labeled_df.index.max()
    start_ts = end_ts - pd.Timedelta(days=days_to_show)

    # --- LP Plot ---
    lp = labeled_df.loc[start_ts:end_ts, lp_col]
    lp_mask = labeled_df.loc[start_ts:end_ts, lp_flag_col]

    plt.figure(figsize=(14, 4))
    plt.plot(lp.index, lp, label=f"{lp_col} (bar)", lw=0.8)
    plt.axhline(lp_thresh, color='red', linestyle='--', label=f"LP low threshold ({lp_thresh} bar)")
    plt.fill_between(lp.index, lp.min(), lp.max(), where=lp_mask, alpha=0.15, color='red', label="Low LP flag")
    plt.ylabel("Pressure (bar)")
    plt.title(f"LP Pressure with Low Flag (last {days_to_show} days)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- HP Plot ---
    hp = labeled_df.loc[start_ts:end_ts, hp_col]
    hp_mask = labeled_df.loc[start_ts:end_ts, hp_flag_col]

    plt.figure(figsize=(14, 4))
    plt.plot(hp.index, hp, label=f"{hp_col} (bar)", lw=0.8)
    plt.axhline(hp_thresh, color='orange', linestyle='--', label=f"HP low threshold ({hp_thresh} bar)")
    plt.fill_between(hp.index, hp.min(), hp.max(), where=hp_mask, alpha=0.15, color='orange', label="Low HP flag")
    plt.ylabel("Pressure (bar)")
    plt.title(f"HP Pressure with Low Flag (last {days_to_show} days)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    #################################################################
    
    
def plot_last_weeks_with_overlays(
    df,
    days=28,
    level_col="HPU_SPLY_LEV_L",
    cons_col="Supply_Consumption_Excl_Fills",
    thr=None,
    show_thresholds=True,
    shade_24h_high=True
):
    """
    Expects columns created by add_pump_states with NEW names:
      HP_pump_rate_2h,  HP_pump_rate_24h,  HP_pump_state
      LP_pump_rate_2h,  LP_pump_rate_24h,  LP_pump_state
    Also uses `is_tank_fill` only if you want to shade elsewhere (not used here).
    """
    # time window
    end   = df.index.max().floor("T")
    start = end - pd.Timedelta(days=days)
    s = df.loc[start:end]

    # overlays (hourly for readability)
    lvl  = s[level_col].resample("1H").mean() if level_col in s.columns else None
    cons = s[cons_col].resample("1H").mean()  if cons_col  in s.columns else None

    cmap = {1:"#2ca02c", 2:"#ff9800", 3:"#d32f2f"}  # steady/elevated/high

    def _panel(prefix, title):
        r2  = f"{prefix}_pump_rate_2h"
        r24 = f"{prefix}_pump_rate_24h"
        st  = f"{prefix}_pump_state"

        # basic presence check
        for col in (r2, r24, st):
            if col not in s.columns:
                raise KeyError(f"Required column missing: {col}")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(s.index, s[r2],  lw=1.1, label=f"{prefix} 2h rate")
        ax.plot(s.index, s[r24], lw=1.1, label=f"{prefix} 24h rate")
        ax.scatter(s.index, s[r2], s=10, c=s[st].map(cmap), zorder=3, label=f"{prefix} state")
        ax.set_ylabel("starts/hour")
        ax.set_title(f"{title} — last {days} days")
        ax.grid(alpha=.3)

        # thresholds & shading if provided
        if thr is not None and show_thresholds and prefix in thr:
            T2e, T2h   = thr[prefix]["2h_elev"], thr[prefix]["2h_high"]
            ax.axhline(T2e, ls="--", alpha=.55, label=f"{prefix} 2h elev")
            ax.axhline(T2h, ls="--", alpha=.55, label=f"{prefix} 2h high")

        if thr is not None and shade_24h_high and prefix in thr:
            T24h = thr[prefix]["24h_high"]
            mask = (s[r24] >= T24h)
            if mask.any():
                ymin, ymax = ax.get_ylim()
                ax.fill_between(s.index, ymin, ymax, where=mask, color="#d32f2f", alpha=0.08, step="pre",
                                label=f"{prefix} 24h≥high")
                ax.set_ylim(ymin, ymax)

        # overlays on right axes
        lines, labels = ax.get_legend_handles_labels()
        if lvl is not None:
            ax2 = ax.twinx()
            ax2.plot(lvl.index, lvl.values, alpha=.75, label="Tank level (L)")
            ax2.set_ylabel("Tank level (L)")
            L2, lab2 = ax2.get_legend_handles_labels()
            lines += L2; labels += lab2

        if cons is not None:
            ax3 = ax.twinx()
            ax3.spines["right"].set_position(("axes", 1.12))
            ax3.plot(cons.index, cons.values, ls="--", alpha=.6, label="Consumption (L/h)")
            ax3.set_ylabel("Consumption (L/h)")
            L3, lab3 = ax3.get_legend_handles_labels()
            lines += L3; labels += lab3

        ax.legend(lines, labels, loc="upper left", ncol=2)
        plt.tight_layout(); plt.show()

    # Make both panels
    _panel("HP", "HP Pump")
    _panel("LP", "LP Pump")

##############################################################

def overlay_external_losses_with_pumps(
    df,
    loss_col='External_Losses',          # L/min time series
    high_mask_col='is_losses_high',      # boolean/0-1 mask from your labeler
    hp_cum_col='Cum_HP_Pump_A_Run_Count',
    lp_cum_col='Cum_LP_Pump_A_Run_Count',
    level_col='HPU_SPLY_LEV_L',          # optional
    window_days=21,                      # last N days
    base_freq='1T',                      # resample base for alignment: '1T', '5T', '1H'
    smooth_minutes=5,                    # smoothing for loss trace (moving mean)
):
    # --- prep & align ---
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.sort_index()

    # base grid for clean alignment
    idx = pd.date_range(df.index.min(), df.index.max(), freq=base_freq)
    loss = df[loss_col].reindex(idx).astype(float)
    mask = df[high_mask_col].reindex(idx).fillna(False).astype(bool)

    # smooth the loss trace a bit so it’s readable
    if base_freq.endswith('T'):
        k = max(1, int(smooth_minutes / int(base_freq[:-1])))
        loss_smooth = loss.rolling(k, min_periods=1).mean()
    else:
        loss_smooth = loss

    # convert cumulative run counts → runs per base_freq → per hour
    def runs_per_hour(cum_series):
        s = df[cum_series].reindex(idx).astype(float).interpolate(limit=2)
        inc = s.diff().clip(lower=0).fillna(0)
        # convert to runs/hour depending on base frequency
        if base_freq.endswith('T'):
            minutes = int(base_freq[:-1])
            return inc * (60 / minutes)
        elif base_freq.endswith('H'):
            hours = int(base_freq[:-1])
            return inc * (1 / hours)
        else:
            return inc  # fallback

    hp_rph = runs_per_hour(hp_cum_col)
    lp_rph = runs_per_hour(lp_cum_col)

    # tank level (optional)
    level = df.get(level_col, pd.Series(index=df.index)).reindex(idx)

    # time window
    end = idx.max()
    start = end - pd.Timedelta(days=window_days)

    tsel = (idx >= start) & (idx <= end)
    t = idx[tsel]

    loss_s = loss_smooth[tsel]
    mask_s = mask[tsel]
    hp_s   = hp_rph[tsel]
    lp_s   = lp_rph[tsel]
    level_s = level[tsel]

    # --- plot ---
    fig, ax1 = plt.subplots(figsize=(14, 4))

    # shaded “high loss” periods
    # draw contiguous spans to keep the plot efficient
    in_span = False
    span_start = None
    for ti, flag in zip(t, mask_s.values):
        if flag and not in_span:
            span_start = ti
            in_span = True
        if in_span and not flag:
            ax1.axvspan(span_start, ti, alpha=0.12)  # light red by default
            in_span = False
    if in_span:
        ax1.axvspan(span_start, t[-1], alpha=0.12)

    ax1.plot(t, loss_s, lw=1.0, label=f'{loss_col} (L/min)')
    ax1.set_ylabel('External loss (L/min)')
    ax1.set_xlabel('date')
    ax1.grid(alpha=0.3)

    # overlay HP/LP runs/hour on same axis (different scale is usually fine),
    # or use a right axis if you prefer separation. Here we keep same axis for quick comparison.
    ax1.plot(t, hp_s, lw=1.0, alpha=0.9, label='HP runs/hour')
    ax1.plot(t, lp_s, lw=1.0, alpha=0.9, label='LP runs/hour')

    # optional: tank level on twin axis
    if level_s.notna().any():
        ax2 = ax1.twinx()
        ax2.plot(t, level_s, lw=1.0, alpha=0.6, linestyle='--', label='Tank level (L)')
        ax2.set_ylabel('Tank level (L)')

        # merge legends
        lines, labels = ax1.get_legend_handles_labels()
        L2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + L2, labels + lab2, loc='upper left', ncol=2)
    else:
        ax1.legend(loc='upper left', ncol=2)

    ax1.set_title('External losses overlaid with pump activity — last {} days'.format(window_days))
    fig.tight_layout()
    plt.show()

    # --- quick co-occurrence stats ---
    pct_time_high = 100 * mask_s.mean()
    pct_high_when_hp = 100 * mask_s[hp_s > 0].mean() if (hp_s > 0).any() else np.nan
    pct_high_when_lp = 100 * mask_s[lp_s > 0].mean() if (lp_s > 0).any() else np.nan
    print(f'% time high losses (window): {pct_time_high:.2f}%')
    if not np.isnan(pct_high_when_hp):
        print(f'% of time high losses while HP running: {pct_high_when_hp:.2f}%')
    if not np.isnan(pct_high_when_lp):
        print(f'% of time high losses while LP running: {pct_high_when_lp:.2f}%')

#####################################
def overlay_losses_with_pump_states(
    df,
    loss_col='External_Losses',
    high_mask_col='is_losses_high',
    hp_state=None,
    lp_state=None,
    window_days=21,
    smooth_minutes=5
):
    """
    Overlay external losses with HP/LP pump states (off/low/medium/high).
    hp_state and lp_state should be pd.Series indexed by datetime with categorical labels.
    """

    # prep external losses
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.sort_index()
    loss = df[loss_col].astype(float)
    mask = df[high_mask_col].astype(bool)

    loss_smooth = loss.rolling(f'{smooth_minutes}T', min_periods=1).mean()

    # time window
    end = df.index.max()
    start = end - pd.Timedelta(days=window_days)

    tsel = (df.index >= start) & (df.index <= end)

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # shaded high-loss spans
    in_span = False
    span_start = None
    for ti, flag in zip(df.index[tsel], mask[tsel].values):
        if flag and not in_span:
            span_start = ti; in_span = True
        if in_span and not flag:
            ax1.axvspan(span_start, ti, alpha=0.12, color='red')
            in_span = False
    if in_span:
        ax1.axvspan(span_start, df.index[tsel][-1], alpha=0.12, color='red')

    # external losses line
    ax1.plot(df.index[tsel], loss_smooth[tsel], lw=1.0, label=f'{loss_col} (L/min)')
    ax1.set_ylabel('External loss (L/min)')
    ax1.set_xlabel('date')
    ax1.grid(alpha=0.3)

    # map states → colors
    cmap = {'off':'grey','steady':'green','elevated':'orange','high':'red'}

    if hp_state is not None:
        st = hp_state.loc[start:end]
        ax1.scatter(st.index, [0.2]*len(st), c=st.map(cmap), s=15, marker='s', label='HP state')

    if lp_state is not None:
        st = lp_state.loc[start:end]
        ax1.scatter(st.index, [-0.2]*len(st), c=st.map(cmap), s=15, marker='s', label='LP state')

    ax1.legend(loc='upper left', ncol=3)
    ax1.set_title(f'External losses with HP/LP pump states — last {window_days} days')

    plt.show()
#state_map = {1: "steady", 2: "elevated", 3: "high"}
#hp_state = labeled_df["HP_pump_state"].map(state_map)
#lp_state = labeled_df["LP_pump_state"].map(state_map)

###############################

def plot_hpu_outputs_with_lp_runs(
    labeled_df: pd.DataFrame,
    window_days: int = 21,
    cum_runs_col: str = "Cum_LP_Pump_A_Run_Count",
    hpu_cols: tuple = ("HPU_LPA_OUT", "HPU_LPB_OUT", "HPU_LP_A_SPLY"),
    shift_minutes: int = 120,
    runs_col_name: str = "LP_Runs_2h",
    modify_inplace: bool = True,
):
    """
    Prepare data (datetime index + sort), compute 2h LP runs from a cumulative counter,
    slice the last `window_days`, and plot HPU signals (left axis) with 2h runs (right axis).

    Parameters
    ----------
    labeled_df : pd.DataFrame
        Time-indexed dataframe containing HPU signals and cumulative run counter.
    window_days : int, default=21
        Number of days from the end to display.
    cum_runs_col : str, default="Cum_LP_Pump_A_Run_Count"
        Column with the cumulative LP pump run count.
    hpu_cols : tuple[str, ...], default=("HPU_LPA_OUT","HPU_LPB_OUT","HPU_LP_A_SPLY")
        HPU columns to plot on the left axis.
    shift_minutes : int, default=120
        Window (in minutes) used to compute runs in the last X minutes.
    runs_col_name : str, default="LP_Runs_2h"
        Name of the derived runs column added to the dataframe.
    modify_inplace : bool, default=True
        If True, modifies `labeled_df` in place; otherwise works on a shallow copy.

    Returns
    -------
    pd.DataFrame
        The sliced dataframe (last `window_days`), including the computed `runs_col_name`.
    """
    # Work in-place or on a shallow copy
    df = labeled_df if modify_inplace else labeled_df.copy(deep=False)

    # --- 1) Ensure datetime index & sort ---
    df.index = pd.to_datetime(df.index, errors="coerce")
    df.sort_index(inplace=True)

    # Validate required columns
    required = set(hpu_cols) | {cum_runs_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    # --- 2) Compute 'runs in last <shift_minutes>' from cumulative counter ---
    # shift by 'shift_minutes' samples (assumes 1-minute sampling);
    # if your sampling isn't 1-minute, replace with shift(periods=...samples)
    runs_series = df[cum_runs_col] - df[cum_runs_col].shift(shift_minutes)
    df[runs_col_name] = runs_series.fillna(0)

    # --- 3) Slice last `window_days` ---
    end = df.index.max()
    start = end - pd.DateOffset(days=window_days)
    df_slice = df.loc[start:end].copy()

    # --- 4) Plot ---
    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Left axis: HPU signals
    for col in hpu_cols:
        ax1.plot(df_slice.index, df_slice[col], label=col, lw=0.9)

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Pressure / Flow")
    ax1.legend(loc="upper left")

    # Right axis: LP runs in last `shift_minutes`
    ax2 = ax1.twinx()
    ax2.plot(
        df_slice.index,
        df_slice[runs_col_name],
        label=f"LP Runs (last {shift_minutes} min)",
        linestyle="--",
        lw=1.0,
    )
    ax2.set_ylabel(f"LP Runs (last {shift_minutes} min)")
    ax2.legend(loc="upper right")

    # Title, grid, date formatting
    plt.title(f"HPU Outputs & LP-Pump-A Run Frequency (last {window_days} days)")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))

    fig.tight_layout()
    plt.show()

    return df_slice
##################################################

def plot_hp_runs(
    labeled_df: pd.DataFrame,
    hp_cum_col: str = "Cum_HP_Pump_A_Run_Count",
    hpu_cols: tuple = ("HPU_HPA_OUT", "HPU_HPB_OUT", "HPU_HP_A_SPLY"),
    windows: dict = None,
    slice_days: int = 1,
    modify_inplace: bool = True,
):
    """
    Compute HP pump run frequencies (2h & 24h by default) from cumulative counter,
    slice the last `slice_days`, and plot HPU signals with run frequencies.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        Time-indexed dataframe with HPU signals and HP pump cumulative run counter.
    hp_cum_col : str, default="Cum_HP_Pump_A_Run_Count"
        Column containing cumulative HP pump run count.
    hpu_cols : tuple[str, ...], default=("HPU_HPA_OUT","HPU_HPB_OUT","HPU_HP_A_SPLY")
        HPU signal columns to plot on the left axis.
    windows : dict, default={"24 h": 1440, "2 h": 120}
        Mapping of label -> shift (in samples) for run frequency calculation.
        Assumes 1-minute sampling. Adjust numbers if sampling differs.
    slice_days : int, default=1
        Number of days from the end to display.
    modify_inplace : bool, default=True
        If True, adds new run-frequency columns to `labeled_df`.

    Returns
    -------
    pd.DataFrame
        The sliced dataframe containing calculated run-frequency columns.
    """
    # Defaults
    if windows is None:
        windows = {"24 h": 1440, "2 h": 120}

    # Work in-place or shallow copy
    df = labeled_df if modify_inplace else labeled_df.copy(deep=False)

    # --- 0) Ensure datetime index & sorted ---
    df.index = pd.to_datetime(df.index, errors="coerce")
    df.sort_index(inplace=True)

    # --- 1) Compute run frequencies for each window ---
    for label, shift in windows.items():
        colname = f"HP_Runs_{label.replace(' ', '')}"
        df[colname] = (df[hp_cum_col] - df[hp_cum_col].shift(shift)).fillna(0)

    # --- 2) Slice last `slice_days` ---
    end = df.index.max()
    start = end - pd.DateOffset(days=slice_days)
    df_slice = df.loc[start:end].copy()

    # --- 3) Plot for each window ---
    for label, shift in windows.items():
        run_col = f"HP_Runs_{label.replace(' ', '')}"

        fig, ax1 = plt.subplots(figsize=(10, 4))

        # Left axis: HPU outputs
        for col in hpu_cols:
            ax1.plot(df_slice.index, df_slice[col], label=col, lw=0.9)

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Pressure / Flow")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))

        # Right axis: HP runs
        ax2 = ax1.twinx()
        ax2.plot(
            df_slice.index,
            df_slice[run_col],
            linestyle="--",
            color="tab:red",
            label=f"HP Runs ({label})",
        )
        ax2.set_ylabel(f"HP Runs in Last {label}")
        ax2.legend(loc="upper right")

        plt.title(f"HPU Outputs & HP-Pump Run Frequency (Last {slice_days} days, Window={label})")
        fig.tight_layout()
        plt.show()

    return df_slice
####################################################################

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_hp_2h_dropslopes(
    labeled_df: pd.DataFrame,
    years: int = 3,
    y_limits: tuple = (-1, 2),
    hpa_col: str = "HPU_HPA_OUT_DropSlope_2h_Lph",
    hpb_col: str = "HPU_HPB_OUT_DropSlope_2h_Lph",
    plot_hpa: bool = True,
    plot_hpb: bool = True,
    separate_figures: bool = True,
):
    """
    Plot 2-hour DropSlope for HPA / HPB over the last `years` years, with y-axis zoom.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        DataFrame with a datetime index (or a 'timestamp' column) and DropSlope columns.
    years : int, default=3
        Window length to display (from the dataset end).
    y_limits : tuple, default=(-1, 2)
        y-axis limits for both plots.
    hpa_col : str
        Column name for HPA 2h DropSlope (Lph).
    hpb_col : str
        Column name for HPB 2h DropSlope (Lph).
    plot_hpa, plot_hpb : bool
        Toggle whether to plot HPA/HPB.
    separate_figures : bool, default=True
        If True, creates separate figures (like your snippet). If False, overlays both on one figure.
    """
    df = labeled_df

    # Ensure datetime index (fallback to 'timestamp' if needed)
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"], errors="coerce"))
        else:
            raise ValueError("DataFrame must have a datetime index or a 'timestamp' column.")

    df = df.sort_index()

    # Time slice
    end_date = df.index.max()
    start_date = end_date - pd.DateOffset(years=years)
    cols_needed = []
    if plot_hpa: cols_needed.append(hpa_col)
    if plot_hpb: cols_needed.append(hpb_col)
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    df_win = df.loc[start_date:end_date, cols_needed]

    def _style_time_axis(ax):
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    if not separate_figures:
        fig, ax = plt.subplots(figsize=(12, 6))
        if plot_hpa:
            ax.plot(df_win.index, df_win[hpa_col], label="HPA DropSlope")
        if plot_hpb:
            ax.plot(df_win.index, df_win[hpb_col], label="HPB DropSlope")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drop Slope (Lph)")
        ax.set_title(f"HPU HPA/HPB 2h Drop Slope (Last {years} Years, Zoomed)")
        ax.legend()
        ax.set_ylim(*y_limits)
        _style_time_axis(ax)
        plt.tight_layout()
        plt.show()
        return

    # Separate figures (mirrors your original)
    if plot_hpa:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_win.index, df_win[hpa_col], label="HPA DropSlope")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drop Slope (Lph)")
        ax.set_title(f"HPU HPA 2h Drop Slope (Last {years} Years, Zoomed)")
        ax.legend()
        ax.set_ylim(*y_limits)
        _style_time_axis(ax)
        plt.tight_layout()
        plt.show()

    if plot_hpb:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_win.index, df_win[hpb_col], label="HPB DropSlope")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drop Slope (Lph)")
        ax.set_title(f"HPU HPB 2h Drop Slope (Last {years} Years, Zoomed)")
        ax.legend()
        ax.set_ylim(*y_limits)
        _style_time_axis(ax)
        plt.tight_layout()
        plt.show()

########################################
def plot_lpa_dropslopes_with_valves(
    labeled_df: pd.DataFrame,
    days: int = 14,
    lpa_2h_col: str = "HPU_LPA_OUT_DropSlope_2h_Lph",
    # lpa_8h_col: str = "HPU_LPA_OUT_DropSlope_8h_Lph",  # keep handy if needed
    valve_5_col: str = "5_LP_Valve_OpenToClosed",
    valve_2_col: str = "2_LP_Valve_OpenToClosed",
    title: str = "HPU_LPA_OUT Drop Slopes & 2″/5″ LP Valve Events",
):
    """
    Plot LPA 2h DropSlope with 2″ and 5″ LP valve Open→Closed event counts on a twin axis.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        DataFrame with a datetime index and the relevant columns.
    days : int, default=14
        Window length to display (from the dataset end).
    lpa_2h_col : str
        Column for LPA 2h DropSlope (L/hour).
    valve_5_col, valve_2_col : str
        Columns for 5″ and 2″ LP valve Open→Closed counts.
    title : str
        Plot title prefix.
    """
    df = labeled_df

    # Ensure datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"], errors="coerce"))
        else:
            raise ValueError("DataFrame must have a datetime index or a 'timestamp' column.")

    df = df.sort_index()

    # Validate columns
    required = [lpa_2h_col, valve_5_col, valve_2_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    # Slice window
    end = df.index.max()
    start = end - pd.Timedelta(days=days)
    df_win = df.loc[start:end]

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df_win.index, df_win[lpa_2h_col], label="2 h DropSlope (L/h)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Drop Slope (L/hour)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.step(df_win.index, df_win[valve_5_col], where="mid", label='5″ LP Valve Open→Closed')
    ax2.step(df_win.index, df_win[valve_2_col], where="mid", label='2″ LP Valve Open→Closed')
    ax2.set_ylabel("Valve Open→Closed Count")
    ax2.legend(loc="upper right")

    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))

    plt.title(f"{title} (Last {days} Days)")
    plt.tight_layout()
    plt.show()

########################################
def plot_drop_slopes(    df,
    loss_col="External_Losses",   # L/min
    slope_cols=[
        "External_Losses_DropSlope_1h_Lph",
        "Supply_Consumption_Excl_Fills_DropSlope_1h_Lph"
    ],
    days=30,
    title="External Losses and Drop Slopes"
):
    """
    Plot External Losses (L/min) together with slope features (L/h).
    External_Losses is plotted on the left y-axis, slopes on the right.
    """

    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors="coerce")
    d = d.sort_index()

    end = d.index.max()
    start = end - pd.Timedelta(days=days)

    d = d.loc[start:end, [loss_col] + slope_cols]

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # --- External losses (L/min) ---
    ax1.plot(d.index, d[loss_col], color="tab:blue", lw=1.0,
             label=f"{loss_col} (L/min)")
    ax1.set_ylabel("External Losses (L/min)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # --- slopes on secondary axis ---
    ax2 = ax1.twinx()
    for col in slope_cols:
        ax2.plot(d.index, d[col], lw=1.0, label=col)
    ax2.set_ylabel("Drop slopes (L/h)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # --- legend ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(f"{title} (last {days} days)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    ########################

def plot_pump_states_timeline(
    df,
    hp_state_col="HP_pump_state",
    lp_state_col="LP_pump_state",
    hp_rate2_col="HP_pump_rate_2h",   # optional; set to None to hide
    lp_rate2_col="LP_pump_rate_2h",
    days=30,
    resample="5T",                    # for cleaner plotting; set None for raw
    title="Pump states (1=steady, 2=elevated, 3=high)"
):
    """
    Plot HP/LP pump state timelines (1,2,3) with optional 2h start-rate overlays.
    Expects a DateTimeIndex and state columns produced by add_pump_states().
    """
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors="coerce")
    d = d[~d.index.isna()].sort_index()

    end = d.index.max()
    start = end - pd.Timedelta(days=days)
    cols = [c for c in [hp_state_col, lp_state_col, hp_rate2_col, lp_rate2_col] if c and c in d.columns]
    d = d.loc[start:end, cols]

    # resample
    if resample:
        agg = {}
        if hp_state_col in d: agg[hp_state_col] = "max"  # any elevated/high -> show it
        if lp_state_col in d: agg[lp_state_col] = "max"
        if hp_rate2_col and hp_rate2_col in d: agg[hp_rate2_col] = "mean"
        if lp_rate2_col and lp_rate2_col in d: agg[lp_rate2_col] = "mean"
        d = d.resample(resample).agg(agg)

    # colors per state
    cmap = {1: "#6BA292",  # green-ish for normal
            2: "#F2C14E",  # amber for elevated
            3: "#D96C6C"}  # red for high

    def _plot_state_strip(ax, s: pd.Series, label: str):
        s = s.astype("float").fillna(1.0)  # default to 1 if missing
        # draw as colored rectangles per run of constant state
        run_id = (s != s.shift()).cumsum()
        for _, grp in s.groupby(run_id):
            state = int(round(grp.iloc[0]))
            ax.axvspan(grp.index[0], grp.index[-1],
                       ymin=0.05, ymax=0.95,
                       color=cmap.get(state, "#CCCCCC"), alpha=0.6)
        ax.set_yticks([1,2,3]); ax.set_ylim(0.5, 3.5)
        ax.set_ylabel(label)
        ax.grid(True, axis="x", alpha=0.3)
        # guide lines
        ax.hlines([1,2,3], xmin=d.index.min(), xmax=d.index.max(), colors="#888888", linestyles=":", linewidth=0.6)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    if hp_state_col in d:
        _plot_state_strip(ax1, d[hp_state_col], "HP state (1/2/3)")
        if hp_rate2_col and hp_rate2_col in d:
            ax1b = ax1.twinx()
            ax1b.plot(d.index, d[hp_rate2_col], lw=1.2, label="HP starts/h (2h window)")
            ax1b.set_ylabel("HP starts/hour (2h)")
            ax1b.legend(loc="upper left")

    if lp_state_col in d:
        _plot_state_strip(ax2, d[lp_state_col], "LP state (1/2/3)")
        if lp_rate2_col and lp_rate2_col in d:
            ax2b = ax2.twinx()
            ax2b.plot(d.index, d[lp_rate2_col], lw=1.2, label="LP starts/h (2h window)")
            ax2b.set_ylabel("LP starts/hour (2h)")
            ax2b.legend(loc="upper left")

    # legend for state colors
    from matplotlib.patches import Patch
    handles = [Patch(color=cmap[1], label="1 = steady/normal"),
               Patch(color=cmap[2], label="2 = elevated"),
               Patch(color=cmap[3], label="3 = high")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)

    fig.suptitle(f"{title} — last {days} days", y=0.98)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
    ##########################################################
    import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_supply_vs_losses(df: pd.DataFrame, window="7D"):
    """Compact line plot: Supply_Consumption_Excl_Fills vs External_Losses, daily totals."""
    d = df.last(window).copy()
    supply = d["Supply_Consumption_Excl_Fills"].resample("D").sum().fillna(0)
    losses = d["External_Losses"].resample("D").sum().fillna(0)

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=110)  # <<< smaller figure
    ax.plot(supply.index, supply.values, label="Supply Consumption (Excl Fills)",
            color="#1f77b4", linewidth=1.8, marker="o", ms=3.5)
    ax.plot(losses.index, losses.values, label="External Losses",
            color="#d62728", linewidth=1.8, marker="s", ms=3.5)

    ax.set_title("Supply vs External Losses — Daily Totals", fontsize=12, weight="bold")
    ax.set_ylabel("Litres/day", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    return fig


def plot_supply_breakdown(df: pd.DataFrame, window="7D"):
    """
    Compact stacked bars: FCV usage, Valve operation, Umbilical charge, Unaccounted
    stacked under Supply_Consumption_Excl_Fills (daily totals).
    """
    d = df.last(window).copy()

    supply = d["Supply_Consumption_Excl_Fills"].resample("D").sum().fillna(0)
    fcv    = d["FCV_Fluid_Usage"].resample("D").sum().fillna(0)
    valves = d["Valve_Operation_Fluid"].resample("D").sum().fillna(0)
    umb    = d["umbilical_charge_volume"].resample("D").sum().fillna(0)

    unaccounted = (supply - (fcv + valves + umb)).clip(lower=0)

    plot_df = pd.DataFrame({
        "FCV Usage": fcv,
        "Valve Operation": valves,
        "Umbilical Charge": umb,
        "Unaccounted": unaccounted
    }, index=supply.index)

    colors = {
        "FCV Usage": "#1f77b4",
        "Valve Operation": "#2ca02c",
        "Umbilical Charge": "#ff7f0e",
        "Unaccounted": "#d62728"
    }

    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=110)  # <<< smaller figure
    bottom = np.zeros(len(plot_df))
    for col in plot_df.columns:
        ax.bar(plot_df.index, plot_df[col].values, bottom=bottom,
               label=col, color=colors[col], width=0.7)
        bottom += plot_df[col].values

    ax.set_title("Supply Consumption Breakdown — Daily Totals", fontsize=12, weight="bold")
    ax.set_ylabel("Litres/day", fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    return fig

    
    
    
    