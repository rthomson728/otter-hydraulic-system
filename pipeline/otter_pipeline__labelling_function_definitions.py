import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def detect_consumption_lessthan_and_stable(
    s: pd.Series,
    *,
    window_minutes: int,
    thresh_enter: float,                 # e.g., 48.0 to tighten
    std_max: float,
    slope_max_lpm_per_hr: float,
    min_stable_run: int,                 # minutes: min continuous True span to keep
    bridge_gap_min: int,                 # minutes: max False gap to bridge
    days_to_show: int | None = None,     # for plotting
    plot: bool = False,
    title: str | None = None,
    return_summary: bool = True
):
 
    # ---- helpers (kept local for easy reuse in a pipeline file) ----
    def rolling_slope_per_hour(y_window: np.ndarray) -> float:
        # slope in L/min per hour, robust to constant windows
        if np.isnan(y_window).any():
            return np.nan
        x = np.arange(len(y_window), dtype=float)
        x -= x.mean()
        y = y_window - y_window.mean()
        denom = (x**2).sum()
        if denom == 0:
            return 0.0
        m_per_min = (x * y).sum() / denom
        return m_per_min * 60.0

    def enforce_min_and_bridge(mask: pd.Series, min_true=30, bridge=3) -> pd.Series:
        m = mask.copy().fillna(False)
        g = (m != m.shift()).cumsum()
        # Bridge short False gaps
        for _, grp in m.groupby(g):
            if grp.iloc[0] is False and len(grp) <= bridge:
                m.loc[grp.index] = True
        # Drop short True runs
        g2 = (m != m.shift()).cumsum()
        for _, grp in m.groupby(g2):
            if grp.iloc[0] is True and len(grp) < min_true:
                m.loc[grp.index] = False
        return m

    def summarize_runs(mask: pd.Series) -> pd.DataFrame:
        m = mask.fillna(False)
        groups = (m != m.shift()).cumsum()
        runs = []
        for _, grp in m.groupby(groups):
            if bool(grp.iloc[0]):
                runs.append((grp.index[0], grp.index[-1], len(grp)))
        if not runs:
            return pd.DataFrame(columns=['start','end','minutes'])
        return pd.DataFrame(runs, columns=['start','end','minutes'])

    def plot_with_state(series: pd.Series, state: pd.Series, title: str, thresh_line: float | None):
        # limit time window if requested
        y = series
        st = state
        if days_to_show is not None:
            end_ts = series.index.max()
            start_ts = end_ts - pd.Timedelta(days=days_to_show)
            y = series.loc[start_ts:end_ts]
            st = state.reindex(y.index).fillna(False)

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(2,1,1)
        ax1.plot(y.index, y.values, linewidth=0.9, label='raw L/min')

        # shade stable spans
        on = (st != st.shift()).cumsum()
        first_label_done = False
        for _, grp in st.groupby(on):
            if grp.iloc[0]:
                ax1.axvspan(grp.index[0], grp.index[-1], alpha=0.15,
                            label=None if first_label_done else 'stable', ymin=0, ymax=1)
                first_label_done = True

        if thresh_line is not None:
            ax1.axhline(thresh_line, linestyle='--', label=f'threshold {thresh_line:.3f}')

        ax1.set_title(title)
        ax1.set_ylabel('L/min')
        ax1.grid(True)
        ax1.legend(loc='upper left')

        # state strip
        ax2 = fig.add_subplot(2,1,2, sharex=ax1)
        ax2.step(st.index, st.astype(int).values, where='post')
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_ylabel('state')
        ax2.set_xlabel('Time')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    # ---- input checks ----
    if not pd.api.types.is_datetime64_any_dtype(s.index):
        raise ValueError("Series index must be datetime-like (one row per minute recommended).")
    s = s.sort_index()

    # ---- rolling features ----
    W = int(window_minutes)
    med = s.rolling(W, min_periods=W, center=True).median()
    std = s.rolling(W, min_periods=W, center=True).std()
    slope_hr = s.rolling(W, min_periods=W, center=True).apply(rolling_slope_per_hour, raw=True)

    # ---- candidate + post-processing ----
    candidate = (med <= float(thresh_enter)) & (std <= float(std_max)) & (slope_hr.abs() <= float(slope_max_lpm_per_hr))
    is_stable = enforce_min_and_bridge(candidate, min_true=int(min_stable_run), bridge=int(bridge_gap_min))
    is_stable = is_stable.reindex(s.index).astype(bool)

    # ---- optional plot ----
    if plot:
        ttl = title or (
            f"Option B — Sliding window {W} min "
            f"(median<{thresh_enter:.3f}, std≤{std_max}, |slope|≤{slope_max_lpm_per_hr} L/min/hr; "
            f"min run {min_stable_run} min, bridge ≤{bridge_gap_min} min)"
        )
        plot_with_state(s, is_stable, ttl, thresh_line=thresh_enter)

    if not return_summary:
        return is_stable

    runs = summarize_runs(is_stable)
    pct = 100.0 * is_stable.fillna(False).mean()
    return is_stable, {"runs": runs, "pct_stable": pct}
