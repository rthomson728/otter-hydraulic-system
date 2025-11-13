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
########################################################################################################################################################################
#Tank/Valve/System Parameters
supply_tank_volume=1000
return_tank_volume=1800
valve5_volume=8 # valve actuation litres 5''
valve2_volume=2 #valve actaution litres 2
valve_hp_volume=1 #22.2cm3 plus line
fcv_step_volume=0.4 #0.7 too mcuh
pcv_step_volume=0.05
hydrostatic_p=18.27 #182m water depth
valve_columns = [
    'P1_PMV', 'P1_PWV', 'P1_AMV', 'P1_SCSSV', 'P1_PDV', 'P1_TDV',
    'P2_PMV', 'P2_PWV', 'P2_AMV', 'P2_SCSSV', 'P2_PDV', 'P2_TDV',
    'P3_PMV', 'P3_PWV', 'P3_AMV', 'P3_SCSSV', 'P3_PDV', 'P3_TDV',
    'I1_PMV', 'I1_PWV', 'I1_AMV',
    'I2_PMV', 'I2_PWV', 'I2_AMV',
    'MPMV_Inlet','Man_CI','SCM1_LP1_COV','SCM1_HP_COV'
]

#Umbicla charge Thresholds for LP and HP Pxs
thresholds = {
    'SCM1_LP_CONS': (30, 190),
    'SCM1_HP_CONS': (30, 450),
}
cov_columns = ["SCM1_LP1_COV", "SCM1_HP_COV","SCM2_LP1_COV", "SCM2_HP_COV"]

# PCV/WCV columns to clean
pcv_columns = ['P1_PCV','P2_PCV','P3_PCV','I1_PCV','I2_PCV']

columns_to_remove = [
    "HPU_SSIV1_OUT", "HPU_SSIV2_OUT", "SCM1_LPA", "SCM1_LPB", "SCM1_HPA", "SCM1_HPB",
    "SCM1_LP_FLOW", "SCM1_HP_FLOW", "SCM1_LP_RET_FLOW", "SCM2_LPA", "SCM2_LPB",
    "SCM2_HPA", "SCM2_HPB", "SCM2_HP_CONS", "SCM2_LP_FLOW", "SCM2_HP_FLOW", "SCM2_LP_RET_FLOW"
]

valve_transition_cols = [
        '2_LP_Valve_OpenToClosed', '2_LP_Valve_ClosedToOpen',
        '5_LP_Valve_OpenToClosed', '5_LP_Valve_ClosedToOpen',
        'HP_Valve_OpenToClosed',  'HP_Valve_ClosedToOpen'
    ]

pump_columns = {
    'LP_Pump_A': 'HPU_LP_A_SPLY',
    'HP_Pump_A': 'HPU_HP_A_SPLY',
}

#File Paths
parquet_path = r"C:\Users\rosst\OneDrive\Control Integrity\Data\Otter 2003 to 2024 with PCV and COV\Parquet_Clean_Output\Otter_All_Combined_PCV.parquet" 

pump_events_path = r"C:\Users\rosst\OneDrive\Control Integrity\otter-hydraulic-system\data\pump_run_events.csv"

# Processd rop lsop deata. This took 36 hourrs to run
drop_slope_data_path= r"C:\Users\rosst\OneDrive\Control Integrity\otter-hydraulic-system\data\Slope_Features_Only.parquet"
SCM_drop_slope_data_path = r"C:\Users\rosst\OneDrive\Control Integrity\otter-hydraulic-system\data\Slope_Features_Only_SCM.parquet"

#For umb charge event sid
thresholds = {
    'SCM1_LP_CONS': (40, 190),    # LPA
    'SCM1_HP_CONS': (40, 450),    # HPA
    'HPU_LPB_OUT': (30, 180),     # LPB pressure indicator
    'HPU_HPB_OUT': (30, 440),     # HPB pressure indicator
    'HPU_LPA_OUT': (20, 180),     # LPB pressure indicator
    'HPU_HPA_OUT': (20, 440),     # HPB pressure indicator
}

valve_state_map = {
    "OPEN": 1,
    "CLOSED": 0,
    "FAULT": -1,
    "No Data": -1,
    "TIMEOUT": -1,
    "UNKNOWN": -1,
}

valve_event_map = {
    "P1_PMV_OpenToClosed": "P1 PMV Closed",
    "P1_PMV_ClosedToOpen": "P1 PMV Opened",
    "P1_PWV_OpenToClosed": "P1 PWV Closed",
    "P1_PWV_ClosedToOpen": "P1 PWV Opened",
    "P1_AMV_OpenToClosed": "P1 AMV Closed",
    "P1_AMV_ClosedToOpen": "P1 AMV Opened",
    "P1_SCSSV_OpenToClosed": "P1 SCSSV Closed",
    "P1_SCSSV_ClosedToOpen": "P1 SCSSV Opened",
    "P1_PDV_OpenToClosed": "P1 PDV Closed",
    "P1_PDV_ClosedToOpen": "P1 PDV Opened",
    "P1_TDV_OpenToClosed": "P1 TDV Closed",
    "P1_TDV_ClosedToOpen": "P1 TDV Opened",
    "P2_PMV_OpenToClosed": "P2 PMV Closed",
    "P2_PMV_ClosedToOpen": "P2 PMV Opened",
    "P2_PWV_OpenToClosed": "P2 PWV Closed",
    "P2_PWV_ClosedToOpen": "P2 PWV Opened",
    "P2_AMV_OpenToClosed": "P2 AMV Closed",
    "P2_AMV_ClosedToOpen": "P2 AMV Opened",
    "P2_SCSSV_OpenToClosed": "P2 SCSSV Closed",
    "P2_SCSSV_ClosedToOpen": "P2 SCSSV Opened",
    "P2_PDV_OpenToClosed": "P2 PDV Closed",
    "P2_PDV_ClosedToOpen": "P2 PDV Opened",
    "P2_TDV_OpenToClosed": "P2 TDV Closed",
    "P2_TDV_ClosedToOpen": "P2 TDV Opened",
    "P3_PMV_OpenToClosed": "P3 PMV Closed",
    "P3_PMV_ClosedToOpen": "P3 PMV Opened",
    "P3_PWV_OpenToClosed": "P3 PWV Closed",
    "P3_PWV_ClosedToOpen": "P3 PWV Opened",
    "P3_AMV_OpenToClosed": "P3 AMV Closed",
    "P3_AMV_ClosedToOpen": "P3 AMV Opened",
    "P3_SCSSV_OpenToClosed": "P3 SCSSV Closed",
    "P3_SCSSV_ClosedToOpen": "P3 SCSSV Opened",
    "P3_PDV_OpenToClosed": "P3 PDV Closed",
    "P3_PDV_ClosedToOpen": "P3 PDV Opened",
    "P3_TDV_OpenToClosed": "P3 TDV Closed",
    "P3_TDV_ClosedToOpen": "P3 TDV Opened",
    "I1_PMV_OpenToClosed": "I1 PMV Closed",
    "I1_PMV_ClosedToOpen": "I1 PMV Opened",
    "I1_PWV_OpenToClosed": "I1 PWV Closed",
    "I1_PWV_ClosedToOpen": "I1 PWV Opened",
    "I1_AMV_OpenToClosed": "I1 AMV Closed",
    "I1_AMV_ClosedToOpen": "I1 AMV Opened",
    "I2_PMV_OpenToClosed": "I2 PMV Closed",
    "I2_PMV_ClosedToOpen": "I2 PMV Opened",
    "I2_PWV_OpenToClosed": "I2 PWV Closed",
    "I2_PWV_ClosedToOpen": "I2 PWV Opened",
    "I2_AMV_OpenToClosed": "I2 AMV Closed",
    "I2_AMV_ClosedToOpen": "I2 AMV Opened",
}
######################################################################################################################################################################


def load_and_clean_otter_data(
    parquet_path: str,
    columns_to_remove: list,
    years_back: int = 8
) -> pd.DataFrame:
    """
    Reads the Otter dataset from a Parquet file, converts & sorts the timestamp,
    filters to the last `years_back` years, renames misnamed columns, drops unused columns,
    and returns the cleaned DataFrame.
    """
    # 1) Read
    df_all_otter = pd.read_parquet(parquet_path)

    # 2) Timestamp → datetime index
    df_all_otter['Timestamp'] = pd.to_datetime(
        df_all_otter['Timestamp'], errors='coerce'
    )
    df_all_otter.set_index('Timestamp', inplace=True)
    df_all_otter.sort_index(inplace=True)

    # 3) Restrict to last N years
    if years_back is not None:
        last_date = df_all_otter.index.max()
        cutoff_date = last_date - pd.DateOffset(years=years_back)
        df_all_otter = df_all_otter.loc[df_all_otter.index >= cutoff_date]

    # 4) Rename misnamed column
    df_all_otter.rename(
        columns={'HPU-RET_LEV': 'HPU_RET_LEV'},
        inplace=True
    )

    # Replace corrupted HPU_RET_LEV with backup if available
    if "HPU-RET_LEV_B" in df_all_otter.columns:
        df_all_otter["HPU_RET_LEV"] = df_all_otter["HPU-RET_LEV_B"]

    # 5) Drop unused columns (only if they exist)
    existing = [c for c in columns_to_remove if c in df_all_otter.columns]
    df_all_otter.drop(columns=existing, inplace=True)

    # 6) Encode SCM crossover valve states (COV)
    cov_encoding_map = {
        "LP2": 2,
        "HP2": 2,
        "LP1": 1,
        "HP1": 1,
    }
    cov_columns = [col for col in df_all_otter.columns if col.endswith('_COV')]
    for col in cov_columns:
        df_all_otter[col] = df_all_otter[col].map(cov_encoding_map).fillna(1).astype(int)

    # 7) Report & return
    if years_back is not None:
        print(f"Loaded {df_all_otter.shape[0]} rows × {df_all_otter.shape[1]} cols "
              f"(last {years_back} years: {cutoff_date.date()} → {last_date.date()})")        
    else:
        print(f"Loaded {df_all_otter.shape[0]} rows × {df_all_otter.shape[1]} cols (all data)")

    return df_all_otter


################################################################################################################################################################

def process_valve_data(df: pd.DataFrame, valve_columns: list) -> pd.DataFrame:
    """
    For each valve in `valve_columns`, adds two columns:
      - <valve>_OpenToClosed
      - <valve>_ClosedToOpen
    Then aggregates into:
      - 2_LP_Valve_OpenToClosed / ClosedToOpen  (all AMVs)
      - 5_LP_Valve_OpenToClosed / ClosedToOpen  (all PMV/PWV/PDV/TDV)
      - HP_Valve_OpenToClosed      / ClosedToOpen  (all SCSSVs)
    """
    # 1) Per‑valve transitions
    for valve in valve_columns:
        prev = df[valve].shift()
        curr = df[valve]
        df[f"{valve}_OpenToClosed"]  = ((prev == 'OPEN')   & (curr == 'CLOSED')).astype(int)
        df[f"{valve}_ClosedToOpen"]  = ((prev == 'CLOSED') & (curr == 'OPEN')).astype(int)

    # 2) Define groups
    amv_valves = [v for v in valve_columns if v.endswith('_AMV')]
    lp_valves  = [v for v in valve_columns if any(v.endswith(s) for s in ('_PMV','_PWV','_PDV','_TDV','Inlet'))]
    hp_valves  = [v for v in valve_columns if v.endswith('_SCSSV')]

    # 3) Combined metrics
    df['2_LP_Valve_OpenToClosed']   = df[[v+'_OpenToClosed' for v in amv_valves]].sum(axis=1)
    df['2_LP_Valve_ClosedToOpen']   = df[[v+'_ClosedToOpen' for v in amv_valves]].sum(axis=1)

    df['5_LP_Valve_OpenToClosed']   = df[[v+'_OpenToClosed' for v in lp_valves ]].sum(axis=1)
    df['5_LP_Valve_ClosedToOpen']   = df[[v+'_ClosedToOpen' for v in lp_valves ]].sum(axis=1)

    df['HP_Valve_OpenToClosed']     = df[[v+'_OpenToClosed' for v in hp_valves ]].sum(axis=1)
    df['HP_Valve_ClosedToOpen']     = df[[v+'_ClosedToOpen' for v in hp_valves ]].sum(axis=1)

    print("✅ Transition tracking columns created and combined metrics computed.")
    return df
################################################################################################################################################################

def valve_fluid_usage_calc(
    df: pd.DataFrame,
    fluid_per_5lp: float,
    fluid_per_2lp: float,
    fluid_per_hp:  float,
    col_5lp: str = '5_LP_Valve_ClosedToOpen',
    col_2lp: str = '2_LP_Valve_ClosedToOpen',
    col_hp:  str = 'HP_Valve_ClosedToOpen'
) -> pd.DataFrame:
    """
    Adds two columns to df:
      - 'Valve_Operation_Fluid': litres used this timestamp
      - 'Cumulative_Valve_Operation_Fluid': running total litres

    """
    df['Valve_Operation_Fluid'] = (
        df[col_5lp] * fluid_per_5lp +
        df[col_2lp] * fluid_per_2lp +
        df[col_hp]  * fluid_per_hp
    )
    df['Cumulative_Valve_Operation_Fluid'] = df['Valve_Operation_Fluid'].cumsum()
    return df
################################################################################################################################################################
def process_fcv(
    df: pd.DataFrame,
    cpi_col: str = 'FCV_CPI',
    step_threshold: float = 1.0,
    fluid_per_step: float = 0.35
) -> pd.DataFrame:
    """
    For a Flow Control Valve CPI series, computes:
      - cumulative full‐step count
      - per‐step flags
      - fluid usage per step
      - cumulative fluid usage

    """
    # if not overridden, pick up the global
    if fluid_per_step is None:
        try:
            fluid_per_step = globals()['fcv_step_volume']
        except KeyError:
            raise ValueError(
                "fcv_step_volume not defined globally; "
                "please pass fluid_per_step explicitly."
            )

    # 1) Ensure numeric CPI and fill gaps
    cpi = df[cpi_col].astype(float).fillna(method='ffill')

    # 2) Δ and flag full steps
    delta = cpi.diff().abs()
    full_step_flag = (delta >= step_threshold).astype(int)

    # 3) Cumulative count of full steps
    df['FCV_CPI_FullSteps'] = full_step_flag.cumsum()

    # 4) Fluid used this timestamp
    df['FCV_FullSteps']    = full_step_flag
    df['FCV_Fluid_Usage']  = full_step_flag * fluid_per_step

    # 5) Cumulative fluid usage
    df['Cumulative_FCV_Fluid_Usage'] = df['FCV_Fluid_Usage'].cumsum()
    
    df.drop(columns='FCV_CALC', inplace=True)

    print("✅ FCV processing complete: full steps & fluid usage added.")
    return df
################################################################################################################################################################

def process_pcv(
    df: pd.DataFrame,
    pcv_columns: list,
    step_threshold: float = 1.0,
    fluid_per_step: float = None
) -> pd.DataFrame:
    """
    For your PCV position series, computes per‐timestamp and cumulative fluid use:
      - PCV_FullSteps:         total # of full steps (abs diff ≥ threshold) this row
      - PCV_Fluid_Usage:       litres used this row (FullSteps × fluid_per_step)
      - Cumulative_PCV_Fluid_Usage: running total litres

    """
    df = df.copy()
    # pick up global if not provided
    if fluid_per_step is None:
        fluid_per_step = globals().get('pcv_step_volume')
        if fluid_per_step is None:
            raise ValueError(
                "Global `pcv_step_volume` not found; "
                "please pass `fluid_per_step` explicitly."
            )
    
    # 1) make sure numeric and forward‐fill gaps
    pcv_vals = df[pcv_columns].astype(float).fillna(method='ffill')
    
    # 2) compute absolute step‐changes
    delta = pcv_vals.diff().abs()
    
    # 3) flag full steps per valve, then sum across all PCVs
    full_steps = (delta >= step_threshold).astype(int)
    df['PCV_FullSteps'] = full_steps.sum(axis=1)
    
    # 4) fluid used this row
    df['PCV_Fluid_Usage'] = df['PCV_FullSteps'] * fluid_per_step
    
    # 5) cumulative fluid usage
    df['Cumulative_PCV_Fluid_Usage'] = df['PCV_Fluid_Usage'].cumsum()
    
    print("✅ PCV processing complete: steps & fluid usage added.")
    return df

################################################################################################################################################################

def convert_tank_levels(
    df: pd.DataFrame,
    supply_pct_cols=('HPU_SPLY_LEV', 'HPU_SPLY_LEV_B'),
    return_pct_cols=('HPU_RET_LEV', 'HPU-RET_LEV_B'),
    supply_volume: float = None,
    return_volume: float = None
) -> pd.DataFrame:

    df = df.copy()

    # pick up globals if not provided
    if supply_volume is None:
        supply_volume = globals().get('supply_tank_volume')
        if supply_volume is None:
            raise ValueError("Global `supply_tank_volume` not found; pass supply_volume explicitly.")
    if return_volume is None:
        return_volume = globals().get('return_tank_volume')
        if return_volume is None:
            raise ValueError("Global `return_tank_volume` not found; pass return_volume explicitly.")

    # convert supply tanks
    for col in supply_pct_cols:
        if col in df.columns:
            df[f"{col}_L"] = df[col] * (supply_volume / 100.0)
        else:
            print(f"⚠️ Supply column '{col}' not found in DataFrame")

    # convert return tanks
    for col in return_pct_cols:
        if col in df.columns:
            df[f"{col}_L"] = df[col] * (return_volume / 100.0)
        else:
            print(f"⚠️ Return column '{col}' not found in DataFrame")

    print("✅ Tank levels converted to litres for:", 
          ", ".join([f"{c}_L" for c in supply_pct_cols + return_pct_cols]))
    return df

################################################################################################################################################################
def detect_fills_safe(
    df: pd.DataFrame,
    level_col: str = "HPU_SPLY_LEV_L",
    min_event_litres: float = 10.0,     # ignore <10 L
    step_noise: float = 0.5,            # ignore tiny wiggles per min
    max_gap: str = "3min",              # bridge short pauses inside a fill
    max_step_litres: float = 150.0,     # sanity cap per minute rise
    smooth_window: str = "5min"  # light median smoothing; None to disable
) -> pd.DataFrame:
    """
    Robust fill detection:
      - deduplicate & regularise to 1-min grid
      - optional light median smoothing
      - clip per-minute rises to max_step_litres
      - sum positive rises within events separated by > max_gap
      - write event volume at end timestamp (0 elsewhere)
    Adds: fill_event_id, tank_fill_volume, tank_fill_cum.
    """
    d = df.copy()

    # --- 0) Ensure datetime index & sort
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index, errors="coerce")
    d = d.sort_index()

    if level_col not in d.columns:
        raise KeyError(f"'{level_col}' not in DataFrame.")

    # --- 1) Deduplicate & regularise to 1-minute grid
    #     (take the last reading per minute, then asfreq to fill missing minutes)
    s = pd.to_numeric(d[level_col], errors="coerce")
    s = s.groupby(s.index.floor("T")).last()              # unique per minute
    s = s.asfreq("1min").interpolate(limit_direction="both")

    # --- 2) Optional light smoothing to suppress spikes
    if smooth_window:
        s = s.rolling(smooth_window, center=True, min_periods=1).median()

    # --- 3) Minute-to-minute diffs; keep only positive rises
    ds = s.diff()
    rises = ds.clip(lower=0.0)
    rises = rises.where(rises >= step_noise, 0.0)
    # clip impossible per-minute increases
    rises = rises.clip(upper=max_step_litres)

    pos = rises[rises > 0]
    if pos.empty:
        out = d.copy()
        out["fill_event_id"] = np.nan
        out["tank_fill_volume"] = 0.0
        out["tank_fill_cum"] = 0.0
        return out

    # --- 4) Group rises into events by time gaps
    max_gap_td = pd.to_timedelta(max_gap)
    new_evt = (pos.index.to_series().diff() > max_gap_td).fillna(True)
    eid = new_evt.cumsum()

    # --- 5) Sum rises per event; drop tiny events
    evt_sum = pos.groupby(eid).sum()
    keep_ids = evt_sum[evt_sum >= min_event_litres].index
    if len(keep_ids) == 0:
        out = d.copy()
        out["fill_event_id"] = np.nan
        out["tank_fill_volume"] = 0.0
        out["tank_fill_cum"] = 0.0
        return out

    pos_kept = pos[eid.isin(keep_ids)]
    kept_ids = eid[eid.isin(keep_ids)]

    # event end timestamp (on the regularised 1-min grid)
    end_ts_by_eid = pos_kept.groupby(kept_ids).apply(lambda s_: s_.index[-1])

    # --- 6) Build outputs aligned to your original dataframe index
    # Start from regularised index, then reindex back to original timestamps (safe).
    tank_fill_volume_reg = pd.Series(0.0, index=s.index)
    for e, t_end in end_ts_by_eid.items():
        tank_fill_volume_reg.loc[t_end] = float(evt_sum.loc[e])

    # Reindex to original df.index (duplicates ok; values will align by timestamp)
    tank_fill_volume = tank_fill_volume_reg.reindex(d.index, method=None).fillna(0.0)

    # event ids only on rising rows (for reference)
    fill_event_id_reg = pd.Series(np.nan, index=s.index)
    fill_event_id_reg.loc[pos_kept.index] = kept_ids.values
    fill_event_id = fill_event_id_reg.reindex(d.index)

    out = d.copy()
    out["fill_event_id"] = fill_event_id
    out["tank_fill_volume"] = tank_fill_volume
    out["tank_fill_cum"] = out["tank_fill_volume"].cumsum()

    return out
################################################################################################################################################################
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
##################################################################################################################################################################

#Only using the supply tank
def system_fluid_consumption(
    df,
    supply_col="HPU_SPLY_LEV_L",
    timestamp_col=None,
    eps=0.0,                 # ignore tiny |ΔSupply| < eps (litres)
    write_rate_col=True,     # also write 'Supply_Consumption_Rate_L_per_h'
):
    """
    Compute system consumption as the negative gradient of Supply only.
    - Only negative changes in Supply are counted as consumption (fills are ignored).
    - Handles irregular timestamps (time-delta aware).
    - Adds/overwrites columns (in-place):
        'Supply_Consumption_Excl_Fills'               [L per step]
        'Cumulative_Supply_Consumption_Excl_Fills'    [L]
        (optional) 'Supply_Consumption_Rate_L_per_h'  [L/hour]
    """

    # --- Ensure datetime index ---
    if not isinstance(df.index, pd.DatetimeIndex):
        if timestamp_col is None:
            for c in ["timestamp","time","date","datetime","Datestamp"]:
                if c in df.columns:
                    timestamp_col = c
                    break
        if timestamp_col is None:
            raise ValueError("Provide a datetime index or `timestamp_col`.")
        df.set_index(pd.to_datetime(df[timestamp_col], utc=True, errors="coerce"), inplace=True)

    df.sort_index(inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    # --- Just use supply ---
    s = df[supply_col].to_numpy(dtype="float64", copy=False)
    total = np.where(np.isfinite(s), s, np.nan)

    # --- Time deltas (seconds) between consecutive points ---
    idx = df.index.view("int64")  # ns since epoch
    dt_sec = np.empty_like(total)
    dt_sec[:] = np.nan
    valid = np.isfinite(total)
    valid_pairs = valid & np.roll(valid, 1)
    dt_sec[valid_pairs] = (idx[valid_pairs] - np.roll(idx, 1)[valid_pairs]) / 1e9  # ns -> s

    # --- Supply change across valid consecutive points ---
    dtotal = np.empty_like(total)
    dtotal[:] = np.nan
    dtotal[valid_pairs] = total[valid_pairs] - np.roll(total, 1)[valid_pairs]

    # --- Optional noise threshold in L ---
    if eps > 0:
        small = np.abs(dtotal) < eps
        dtotal[small] = 0.0

    # --- Consumption as negative gradient of Supply ---
    consumption_step_L = np.maximum(0.0, -np.nan_to_num(dtotal, nan=0.0))

    # Optional rate (L/hour), time-aware
    if write_rate_col:
        rate = np.full_like(consumption_step_L, np.nan, dtype="float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            rate_vals = consumption_step_L / (dt_sec / 3600.0)
        rate[(dt_sec > 0)] = rate_vals[(dt_sec > 0)]
        df["Supply_Consumption_Rate_L_per_h"] = rate

    # Cumulative litres consumed
    cum_consumption = np.nancumsum(consumption_step_L)

    # --- Write results ---
    df["Supply_Consumption_Excl_Fills"] = consumption_step_L
    df["Cumulative_Supply_Consumption_Excl_Fills"] = cum_consumption

    return df

################################################################################################################################################################

#includes the return tank
def system_fluid_consumption_v2(
    df,
    supply_col="HPU_SPLY_LEV_L_Smooth",
    return_col="HPU_RET_LEV_L_Smooth",
    timestamp_col=None,
    eps=0.0,                 # ignore tiny |ΔTotal| < eps (litres)
    write_rate_col=True,     # also write 'Supply_Consumption_Rate_L_per_h'
):
    """
    Compute system consumption as the negative gradient of Total = Supply + Return.
    - Only negative changes in Total are counted as consumption (fills are ignored).
    - Handles irregular timestamps (time-delta aware).
    - Adds/overwrites columns (in-place):
        'Supply_Consumption_Excl_Fills'               [L per step]
        'Cumulative_Supply_Consumption_Excl_Fills'    [L]
        (optional) 'Supply_Consumption_Rate_L_per_h'  [L/hour]
    """

    # --- Ensure datetime index ---
    if not isinstance(df.index, pd.DatetimeIndex):
        if timestamp_col is None:
            for c in ["timestamp","time","date","datetime","Datestamp"]:
                if c in df.columns:
                    timestamp_col = c
                    break
        if timestamp_col is None:
            raise ValueError("Provide a datetime index or `timestamp_col`.")
        df.set_index(pd.to_datetime(df[timestamp_col], utc=True, errors="coerce"), inplace=True)

    df.sort_index(inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    # --- Build Total where both are finite ---
    s = df[supply_col].to_numpy(dtype="float64", copy=False)
    r = df[return_col].to_numpy(dtype="float64", copy=False)
    total = np.where(np.isfinite(s) & np.isfinite(r), s + r, np.nan)

    # --- Time deltas (seconds) between consecutive points ---
    # Use NaN where either current or previous total is NaN (don't count across gaps)
    idx = df.index.view("int64")  # ns since epoch
    dt_sec = np.empty_like(total)
    dt_sec[:] = np.nan
    valid = np.isfinite(total)
    valid_pairs = valid & np.roll(valid, 1)
    dt_sec[valid_pairs] = (idx[valid_pairs] - np.roll(idx, 1)[valid_pairs]) / 1e9  # ns -> s

    # --- Total change across valid consecutive points ---
    dtotal = np.empty_like(total)
    dtotal[:] = np.nan
    dtotal[valid_pairs] = total[valid_pairs] - np.roll(total, 1)[valid_pairs]

    # --- Optional noise threshold in L ---
    if eps > 0:
        small = np.abs(dtotal) < eps
        dtotal[small] = 0.0

    # --- Consumption as negative gradient of Total ---
    # Per step litres consumed (positive when Total decreases)
    consumption_step_L = np.maximum(0.0, -np.nan_to_num(dtotal, nan=0.0))

    # Optional rate (L/hour), time-aware
    if write_rate_col:
        rate = np.full_like(consumption_step_L, np.nan, dtype="float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            rate_vals = consumption_step_L / (dt_sec / 3600.0)
        # Only keep where dt>0; else NaN
        rate[(dt_sec > 0)] = rate_vals[(dt_sec > 0)]
        df["Supply_Consumption_Rate_L_per_h"] = rate

    # Cumulative litres consumed
    cum_consumption = np.nancumsum(consumption_step_L)

    # --- Write results ---
    df["Supply_Consumption_Excl_Fills"] = consumption_step_L
    df["Cumulative_Supply_Consumption_Excl_Fills"] = cum_consumption

    return df

#################################################################################################################

def add_external_losses(
    df: pd.DataFrame,
    total_col: str = 'Supply_Consumption_Excl_Fills',
    fcv_col:   str = 'FCV_Fluid_Usage',
    valve_col: str = 'Valve_Operation_Fluid',
    pcv_col:   str = 'PCV_Fluid_Usage',
    umb_col:   str = 'umbilical_charge_volume',
    loss_col:  str = 'External_Losses'
) -> pd.DataFrame:
    """
    Adds a column for external fluid losses at the native frequency:
      External_Losses = Total 
                      - FCV 
                      - Valve 
                      - PCV 
                      - Umbilical
    
    Parameters:
      df        : minute‑resolution DataFrame
      total_col : name of total consumption column
      fcv_col   : name of FCV fluid usage column
      valve_col : name of valve operation fluid column
      pcv_col   : name of PCV fluid usage column
      umb_col   : name of umbilical charge rate column
      loss_col  : name for the new loss column
    
    Returns:
      The same DataFrame, with `loss_col` added.
    """
    df = df.copy()
    df[loss_col] = (
        df[total_col]
      - df[fcv_col]
      - df[valve_col]
      - df[pcv_col]
      - df[umb_col]
    )
    return df

################################################################################################################################################################

def add_external_loss_moving_averages(
    df: pd.DataFrame,
    loss_col: str = 'External_Losses',
    windows: list = [2, 12, 24, 168]
) -> pd.DataFrame:
    """
    Computes simple moving averages of your external losses over various
    hourly windows, and adds one column per window:
       MA_{w}h
    
    Uses a time‑based rolling window so that each timestamp’s average covers
    the preceding w hours.

    Parameters:
      df       : DataFrame with a datetime index and `loss_col` present
      loss_col : name of the column to average
      windows  : list of window‑sizes in hours
    
    Returns:
      The same DataFrame, with columns MA_2h, MA_12h, etc. added.
    """
    df = df.copy()
    for w in windows:
        ma_col = f"MA_{w}h"
        df[ma_col] = (
            df[loss_col]
              .rolling(window=f"{w}H", min_periods=1)
              .mean()
        )
    return df

################################################################################################################################################################
def add_hourly_external_loss_mas(
    df: pd.DataFrame,
    loss_col: str = 'External_Losses',
    windows: list = [2, 12, 24, 168]
) -> pd.DataFrame:
    """
    Reproduces the style of MAs you charted:
      - Hourly‐sum External_Losses (L/hour)
      - MA_{w}h = integer‐window mean over those hourly sums

    Returns the hourly DataFrame with columns:
      External_Losses, MA_2h, MA_12h, MA_24h, MA_168h
    """
    # 1) Hourly sums
    hourly = df[loss_col].resample('H').sum().to_frame()

    # 2) Integer‐count rolling on the hourly sums
    for w in windows:
        hourly[f"MA_{w}h"] = (
            hourly['External_Losses']
                  .rolling(window=w, min_periods=1)
                  .mean()
        )
    return hourly
################################################################################################################################################################
def add_daily_ewm_to_minutely_df(
    df: pd.DataFrame,
    loss_col: str = 'External_Losses',
    spans: list = [1, 7, 30]
) -> pd.DataFrame:
    """
    Computes daily EWMs of your loss series and merges them back onto
    the original minute‑resolution DataFrame, forward‑filling each day’s
    EWM value across all minutes of that day.

    Parameters:
      df       : minute‑resolution DataFrame with datetime index
      loss_col : name of the minute‑level loss column (L/min)
      spans    : list of spans in days for the EWMs

    Returns:
      A new DataFrame with the original columns plus one column
      EWM_{span}d (L/day) for each span in spans.
    """
    # 1) Build daily total (L/day)
    daily = (
        df[loss_col]
          .resample('D')
          .sum()
          .to_frame(name=f"{loss_col}_per_day")
    )
    
    # 2) Compute EWMs on that daily series
    for span in spans:
        daily[f"EWM_{span}d"] = (
            daily[f"{loss_col}_per_day"]
              .ewm(span=span, adjust=False)
              .mean()
        )
    
    # 3) Reindex daily EWMs to the minute index, forward‑fill each day
    #    Drop the per_day column if you don’t need it on the minute df.
    ewms = daily[[f"EWM_{span}d" for span in spans]]\
             .reindex(df.index, method='ffill')
    
    # 4) Join back onto the original
    df_out = df.copy()
    for span in spans:
        df_out[f"EWM_{span}d"] = ewms[f"EWM_{span}d"]
    
    return df_out
################################################################################################################################################################

def compute_daily_ewm(
    df: pd.DataFrame,
    loss_col: str = 'External_Losses',
    spans: list = [1, 7, 30]
) -> pd.DataFrame:
    """
    From your minute‐level df, builds a daily loss series and adds
    exponentially‐weighted moving averages.

    Parameters:
      df       : original DataFrame with a datetime index
      loss_col : column name of the minute‐level loss series (L/min)
      spans    : list of spans in days for EWM (e.g. [1,7,30])

    Returns:
      daily_ewm: DataFrame indexed by day with columns:
        - {loss_col}_per_day
        - EWM_{span}d  for each span in spans
    """
    # 1) Make sure index is datetime and sorted
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors='coerce')
    df.sort_index(inplace=True)

    # 2) Build daily total (L/day)
    daily = (
        df[loss_col]
          .resample('D')
          .sum()
          .to_frame(name=f"{loss_col}_per_day")
    )

    # 3) Compute EWMs
    for span in spans:
        ewm_col = f"EWM_{span}d"
        daily[ewm_col] = (
            daily[f"{loss_col}_per_day"]
              .ewm(span=span, adjust=False)
              .mean()
        )

    return daily
################################################################################################################################################################
import warnings
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

def compute_baseline_drift(
    df: pd.DataFrame,
    lp_col: str = 'SCM1_LP_CONS',
    hp_col: str = 'SCM1_HP_CONS',
    tank_col: str = 'HPU_SPLY_LEV_L_Smooth',
    fcv_steps_col: str = 'FCV_FullSteps',
    valve_transition_cols: list = (
        '2_LP_Valve_OpenToClosed', '2_LP_Valve_ClosedToOpen',
        '5_LP_Valve_OpenToClosed', '5_LP_Valve_ClosedToOpen',
        'HP_Valve_OpenToClosed',  'HP_Valve_ClosedToOpen'
    ),
    min_lp: float = 180,
    min_hp: float = 200,
    clean_duration: pd.Timedelta = pd.Timedelta('12h')
) -> (pd.DataFrame, float):
    """
    Returns:
        baseline_df: DataFrame with ['start','end','drop_L'] (may be empty)
        slope_lday : float (may be np.nan), baseline drift in L/day
    Never returns None.
    """
    empty = pd.DataFrame(columns=['start','end','drop_L'])
    try:
        d = df.copy()

        # --- index hygiene (+ dedupe reduces ill-conditioning) ---
        d.index = pd.to_datetime(d.index, errors='coerce')
        d = d[~d.index.isna()].sort_index()
        if d.index.has_duplicates:
            d = d[~d.index.duplicated(keep='last')]

        # --- required columns present? ---
        for c_needed in [lp_col, hp_col, tank_col]:
            if c_needed not in d.columns:
                out = empty.copy()
                out.attrs['error'] = f"Missing required column: {c_needed}"
                return out, np.nan

        # --- numeric casts ---
        d['LP']   = pd.to_numeric(d[lp_col], errors='coerce')
        d['HP']   = pd.to_numeric(d[hp_col], errors='coerce')
        d['Tank'] = pd.to_numeric(d[tank_col], errors='coerce')
        if fcv_steps_col in d.columns:
            d['FCV_steps'] = pd.to_numeric(d[fcv_steps_col], errors='coerce').fillna(0).astype('int64')
        else:
            d['FCV_steps'] = 0

        # --- valve transition columns (missing -> 0) ---
        vcols = list(valve_transition_cols) if valve_transition_cols is not None else []
        for c in vcols:
            if c not in d.columns:
                d[c] = 0
            else:
                d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0).astype('int64')

        # --- clean-period mask (strict) ---
        mask = (
            (d['LP'] > min_lp) &
            (d['HP'] > min_hp) &
            (d['FCV_steps'] == 0) &
            (d[vcols].sum(axis=1) == 0 if vcols else True)
        ).fillna(False)

        # --- contiguous runs ---
        d['run_id'] = (mask != mask.shift(fill_value=False)).cumsum()

        # helper: last known value <= ts (robust to duplicates)
        def value_at_end(series: pd.Series, ts) -> float:
            ts = pd.to_datetime(ts, errors='coerce')
            if pd.isna(ts):
                return np.nan
            s = pd.to_numeric(series, errors='coerce')
            s = pd.Series(s.values, index=pd.to_datetime(series.index, errors='coerce'))
            s = s[~s.index.duplicated(keep='last')].sort_index()
            if s.empty or ts < s.index[0]:
                return np.nan
            pos = s.index.searchsorted(ts, side='right') - 1
            if pos < 0:
                return np.nan
            val = s.iloc[pos]
            return float(val) if np.isfinite(val) else np.nan

        # --- pick one 12h window per clean run (your original logic) ---
        periods = []
        for _, grp in d[mask].groupby('run_id'):
            if grp.empty:
                continue
            start = grp.index[0]
            end   = start + clean_duration
            if end <= grp.index[-1]:
                lvl0 = d.at[start, 'Tank'] if start in d.index else np.nan
                lvl1 = value_at_end(d['Tank'], end)
                if np.isfinite(lvl0) and np.isfinite(lvl1):
                    periods.append({'start': start, 'end': end, 'drop_L': float(lvl0 - lvl1)})

        baseline_df = pd.DataFrame(periods)
        if baseline_df.empty:
            return empty.copy(), np.nan

        # --- numerically-stable linear fit of drop vs start time (L/day) ---
        x_raw = mdates.date2num(pd.to_datetime(baseline_df['start']))
        y_raw = pd.to_numeric(baseline_df['drop_L'], errors='coerce').astype('float64').to_numpy()
        x_raw = np.asarray(x_raw, dtype='float64')

        good = np.isfinite(x_raw) & np.isfinite(y_raw)
        x = x_raw[good]; y = y_raw[good]

        if x.size >= 2:
            x_mean = float(np.mean(x))
            x_std  = float(np.std(x))
            if x_std > 0:
                x_norm = (x - x_mean) / x_std
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", np.RankWarning)
                    slope_norm, _ = np.polyfit(x_norm, y, 1)
                slope_lday = float(slope_norm / x_std)  # convert back to L/day
            else:
                slope_lday = 0.0
        else:
            slope_lday = 0.0

        return baseline_df, slope_lday

    except Exception as e:
        out = empty.copy()
        out.attrs['error'] = f"{type(e).__name__}: {e}"
        return out, np.nan



################################################################################################################################################################
def add_baseline_columns(df: pd.DataFrame, **drift_kwargs) -> pd.DataFrame:
    baseline_df, slope_lday = compute_baseline_drift(df, **drift_kwargs)

    drop_map = {}
    if isinstance(baseline_df, pd.DataFrame) and not baseline_df.empty:
        drop_map = dict(zip(baseline_df['start'], baseline_df['drop_L']))

    df_out = df.copy()
    # map known drops to their window-start timestamps; others NaN
    df_out['baseline_drop_L'] = df_out.index.map(drop_map)
    # constant slope across all rows (may be NaN if no info)
    df_out['baseline_drift_L_per_day'] = (
        float(slope_lday) if (slope_lday is not None and np.isfinite(slope_lday)) else np.nan
    )

    # OPTIONAL: keep last error message from compute in attrs for downstream logging
    if isinstance(baseline_df, pd.DataFrame) and 'error' in getattr(baseline_df, 'attrs', {}):
        df_out.attrs['baseline_error'] = baseline_df.attrs['error']

    return df_out
################################################################################################################################################################
def add_pressure_deltas(
    df: pd.DataFrame,
    lpa_out_col:    str = 'HPU_LPA_OUT',
    lp_cons_col:    str = 'SCM1_LP_CONS',
    hpa_out_col:    str = 'HPU_HPA_OUT',
    hp_cons_col:    str = 'SCM1_HP_CONS'
) -> pd.DataFrame:
    """
    Return a copy of df with two new columns:
      - LP_Px_Delta = lpa_out_col - lp_cons_col
      - HP_Px_Delta = hpa_out_col - hp_cons_col
    """
    df = df.copy()
    
    # ensure numeric
    df[lpa_out_col] = pd.to_numeric(df[lpa_out_col], errors='coerce')
    df[lp_cons_col] = pd.to_numeric(df[lp_cons_col], errors='coerce')
    df[hpa_out_col] = pd.to_numeric(df[hpa_out_col], errors='coerce')
    df[hp_cons_col] = pd.to_numeric(df[hp_cons_col], errors='coerce')
    
    # compute deltas
    df['LP_Px_Delta'] = df[lpa_out_col] - df[lp_cons_col]
    df['HP_Px_Delta'] = df[hpa_out_col] - df[hp_cons_col]
    
    return df

################################################################################################################################################################
def add_pump_run_cumulatives(
    df_all_otter: pd.DataFrame,
    pump_run_csv: str,
    pumps=("LP_Pump_A", "HP_Pump_A"),
    count_col_template="Cum_{pump}_Run_Count",
    dur_col_template="Cum_{pump}_Run_Dur",
    time_col="Start Time",
    duration_col="Duration (min)",
    freq="T"
):
    """
    Reads pump‐run events from CSV and adds cumulative columns, restricting
    the source events to df's time window for efficiency.
    """
    if df_all_otter.empty:
        return df_all_otter

    # 1) Prepare main DF
    df = df_all_otter.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    start, end = df.index.min(), df.index.max()

    # 2) Load runs CSV
    runs = pd.read_csv(pump_run_csv, parse_dates=[time_col])

    # 3) Clean + window the runs
    runs = runs.copy()
    runs[time_col] = pd.to_datetime(runs[time_col], errors='coerce')
    runs = runs[~runs[time_col].isna()]
    # keep only runs whose start time falls within df's window
    runs = runs[(runs[time_col] >= start) & (runs[time_col] <= end)]

    # 4) For each pump, build & reindex cum‐count & cum‐dur
    for pump in pumps:
        rp = runs[runs["Pump"] == pump]

        if rp.empty:
            # just fill zeros for the window
            df[count_col_template.format(pump=pump)] = 0.0
            df[dur_col_template.format(pump=pump)] = 0.0
            continue

        ts_count = rp.set_index(time_col).resample(freq).size().cumsum()
        ts_dur   = rp.set_index(time_col)[duration_col].resample(freq).sum().cumsum()

        df[count_col_template.format(pump=pump)] = (
            ts_count.reindex(df.index, method="ffill").fillna(0)
        )
        df[dur_col_template.format(pump=pump)] = (
            ts_dur.reindex(df.index, method="ffill").fillna(0)
        )

    return df

################################################################################################################################################################
def add_slope_features(
    df_all_otter: pd.DataFrame,
    slope_parquet_path: str
) -> pd.DataFrame:
    """
    Read slope features from a Parquet file, restrict to df's time window,
    and merge onto df_all_otter.
    """
    if df_all_otter.empty:
        return df_all_otter

    # --- main df window ---
    df_all_otter = df_all_otter.copy()
    df_all_otter.index = pd.to_datetime(df_all_otter.index, errors="coerce")
    df_all_otter.sort_index(inplace=True)
    start, end = df_all_otter.index.min(), df_all_otter.index.max()

    # 1) Load slopes
    slope_df = pd.read_parquet(slope_parquet_path)

    # 2) Ensure datetime index & sorted
    slope_df.index = pd.to_datetime(slope_df.index, errors="coerce")
    slope_df = slope_df[~slope_df.index.isna()].sort_index()

    # 2.5) Restrict to df window (auto-trim)
    slope_df = slope_df.loc[(slope_df.index >= start) & (slope_df.index <= end)]

    # 3) Drop duplicate timestamps
    before = len(slope_df)
    slope_df = slope_df[~slope_df.index.duplicated(keep='first')]
    dropped = before - len(slope_df)
    if dropped:
        print(f"Dropped {dropped} duplicate rows from slope_df.")

    # 4) Select only drop-slope related columns for LP and HP
    # 
    drop_slope_cols = ['Slope_2H_Lph', 'Slope_12H_Lph', 'Slope_24H_Lph', 'Slope_1H_Lph', 'Slope_7D_Lph',
        'HPU_LPA_OUT_DropSlope_1h_Lph','HPU_LPA_OUT_DropSlope_2h_Lph','HPU_LPA_OUT_DropSlope_8h_Lph',
        'HPU_LPB_OUT_DropSlope_1h_Lph','HPU_LPB_OUT_DropSlope_2h_Lph','HPU_LPB_OUT_DropSlope_8h_Lph',
        'HPU_HPA_OUT_DropSlope_1h_Lph','HPU_HPA_OUT_DropSlope_2h_Lph','HPU_HPA_OUT_DropSlope_8h_Lph',
        'HPU_HPB_OUT_DropSlope_1h_Lph','HPU_HPB_OUT_DropSlope_2h_Lph','HPU_HPB_OUT_DropSlope_8h_Lph'
    ]
    slope_df = slope_df[[c for c in drop_slope_cols if c in slope_df.columns]]

    # 5) Align index names & reset for merge
    slope_df.index.name = 'timestamp'
    df_all_otter.index.name = 'timestamp'
    df_all_otter = df_all_otter.reset_index()
    slope_df = slope_df.reset_index()

    # 6) Drop overlapping (non-timestamp) columns if needed
    overlapping = df_all_otter.columns.intersection(slope_df.columns)
    overlapping = overlapping.drop('timestamp', errors='ignore')
    if not overlapping.empty:
        print(f"Removing overlapping columns from df_all_otter: {list(overlapping)}")
        df_all_otter = df_all_otter.drop(columns=overlapping)

    # 7) Merge and restore datetime index
    df_aug = pd.merge(df_all_otter, slope_df, how='left', on='timestamp')
    df_aug.set_index('timestamp', inplace=True)
    df_aug.index = pd.to_datetime(df_aug.index, errors='coerce')
    df_aug.sort_index(inplace=True)
    df_aug.index.name = None

    # 8) Quick report
    if slope_df.empty:
        print("Warning: no slope rows within df's time window; slope columns will be NaN.")
    else:
        missing = df_aug[[c for c in drop_slope_cols if c in df_aug.columns]].isnull().all(axis=1).sum()
        if missing:
            print(f"Warning: {missing} df timestamps had no slope data (NaN inserted).")

    return df_aug
##################################################################################################################################################################
def tv1d_denoise(y, lam): #smooth the tanklvels helper
    n = len(y)
    if n == 0 or lam <= 0:
        return y.astype(float).copy()
    x = np.empty(n, dtype=float)
    k = k0 = 0
    vmin = y[0] - lam
    vmax = y[0] + lam
    umin = lam
    umax = -lam
    for i in range(1, n):
        d = y[i] - y[i-1]
        umin += d; umax += d
        if umin > lam:
            while k <= i-1: x[k] = vmin; k += 1
            k0 = i; vmin = y[i] - lam; vmax = y[i] + lam; umin = lam; umax = -lam
        elif umax < -lam:
            while k <= i-1: x[k] = vmax; k += 1
            k0 = i; vmin = y[i] - lam; vmax = y[i] + lam; umin = lam; umax = -lam
        else:
            if umin < -lam:
                vmin += (umin + lam) / (i - k0 + 1); umin = -lam
            if umax > lam:
                vmax += (umax - lam) / (i - k0 + 1); umax = lam
    vbar = vmin + umin / (n - k0 + 1)
    while k <= n - 1: x[k] = vbar; k += 1
    return x
###########################################################################
def add_smoothed_supply_level_litres(
    df: pd.DataFrame,
    a_L: str = "HPU_SPLY_LEV_L",
    b_L: str = "HPU_SPLY_LEV_B_L",
    combine: str = "mean",          # 'min', 'mean', or 'both' (alias to 'min')
    strength: float = 4.8,         # higher -> flatter plateaus
    use_skimage: bool = True,
    out_col: str = "HPU_SPLY_LEV_L_Smooth"
) -> pd.DataFrame:
    """
    Create a smoothed litres column from redundant supply tank sensors.
    - Handles single/missing sensors
    - Fills NaNs before denoising (TV doesn't accept NaN)
    - Never imports inside the function (avoids UnboundLocalError)
    """
    d = df.copy()

    # Accept alias
    if combine == "both":
        combine = "min"   # conservative; change to 'mean' if you prefer

    # Collect available sensors (don't hard-fail if B is missing)
    cols = [c for c in (a_L, b_L) if c in d.columns]
    if not cols:
        raise KeyError(f"Required column '{a_L}' (or '{b_L}') not found. Run convert_tank_levels first.")

    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Combine with skipna so one good sensor still yields a value
    if combine == "min":
        raw = d[cols].min(axis=1, skipna=True)
    elif combine == "mean":
        raw = d[cols].mean(axis=1, skipna=True)
    else:
        raise ValueError("combine must be 'min', 'mean', or 'both'.")

    # If everything is NaN, just return copy of primary (still NaN-safe)
    if raw.notna().sum() == 0:
        d[out_col] = pd.to_numeric(d.get(a_L), errors="coerce")
        return d

    # Fill NaNs before denoise
    raw_filled = raw.interpolate(method="time", limit_direction="both")
    y = raw_filled.to_numpy(dtype=float)

    # Denoise
    y_smooth = None
    if use_skimage:
        try:
            from skimage.restoration import denoise_tv_chambolle
            y_smooth = denoise_tv_chambolle(y, weight=strength, channel_axis=None)
        except Exception:
            pass
    if y_smooth is None:
        # fallback to TV-1D if available, else EWMA
        try:
            y_smooth = tv1d_denoise(y, lam=strength)  # your helper
        except Exception:
            y_smooth = pd.Series(y, index=raw.index).ewm(span=60, adjust=False).mean().to_numpy()

    # If any NaNs slipped through, patch with EWMA at those positions
    if np.isnan(y_smooth).any():
        ys = pd.Series(y_smooth, index=raw.index)
        ys = ys.fillna(ys.ewm(span=60, adjust=False).mean())
        y_smooth = ys.to_numpy()

    d[out_col] = pd.Series(y_smooth, index=raw.index)
    return d
####################################################################################################################################################################

def add_smoothed_return_level_litres(
    df: pd.DataFrame,
    a_L: str = "HPU_RET_LEV_L",
    b_L: str = "HPU_RET_LEV_B_L",      # alt sometimes: "HPU-RET_LEV_B_L"
    combine: str = "min",              # 'min', 'mean', or 'both' (alias to 'min')
    strength: float = 4.8,             # higher -> flatter plateaus
    use_skimage: bool = True,
    out_col: str = "HPU_RET_LEV_L_Smooth"
) -> pd.DataFrame:
    d = df.copy()

    # Accept alias
    if combine == "both":
        combine = "min"

    # Gather available sensors (support hyphenated alt)
    cols = []
    if a_L in d.columns:
        cols.append(a_L)
    if b_L in d.columns:
        cols.append(b_L)
    elif "HPU-RET_LEV_B_L" in d.columns:   # alternate naming seen in your logs
        cols.append("HPU-RET_LEV_B_L")

    if not cols:
        raise KeyError(f"Required column '{a_L}' (or '{b_L}') not found. Run convert_tank_levels first.")

    # Coerce numeric
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Combine sensors with NaN tolerance
    if combine == "min":
        raw = d[cols].min(axis=1, skipna=True)
    elif combine == "mean":
        raw = d[cols].mean(axis=1, skipna=True)
    else:
        raise ValueError("combine must be 'min', 'mean', or 'both'.")

    # If nothing usable, just return the primary as-is (NaN-safe)
    if raw.notna().sum() == 0:
        d[out_col] = pd.to_numeric(d.get(a_L), errors="coerce")
        return d

    # Fill NaNs BEFORE denoising (TV denoise can't handle NaNs)
    raw_filled = raw.interpolate(method="time", limit_direction="both")
    y = raw_filled.to_numpy(dtype=float)

    # Denoise: skimage TV → tv1d fallback → EWMA last-resort
    y_smooth = None
    if use_skimage:
        try:
            from skimage.restoration import denoise_tv_chambolle
            y_smooth = denoise_tv_chambolle(y, weight=strength, channel_axis=None)
        except Exception:
            y_smooth = None
    if y_smooth is None:
        try:
            y_smooth = tv1d_denoise(y, lam=strength)  # if you have this helper
        except Exception:
            y_smooth = pd.Series(y, index=raw.index).ewm(span=60, adjust=False).mean().to_numpy()

    # Patch any residual NaNs from the denoiser
    if np.isnan(y_smooth).any():
        ys = pd.Series(y_smooth, index=raw.index)
        ys = ys.fillna(ys.ewm(span=60, adjust=False).mean())
        y_smooth = ys.to_numpy()

    d[out_col] = pd.Series(y_smooth, index=raw.index)
    return d
#######################################################################
def add_pump_runtime_metrics(df: pd.DataFrame,
                             run_col: str = "Cum_LP_Pump_A_Run_Count",
                             prefix: str = "LP") -> pd.DataFrame:
    """
    Adds pump run frequency and cumulative runtime to the DataFrame.

    Creates:
      - {prefix}_Runs_24h : number of pump runs in the last 24h
      - {prefix}_Runtime_24h_min : runtime in minutes (1 min per run)
      - {prefix}_Runtime_Cumulative_min : cumulative runtime in minutes
    """
    d = df.copy()
    d.index = pd.to_datetime(d.index, errors='coerce')
    d.sort_index(inplace=True)

    # 24h rolling runs
    d[f"{prefix}_Runs_24h"] = (
        d[run_col] - d[run_col].shift(1440)   # 1440 minutes = 24h
    ).clip(lower=0).fillna(0)

    # Runtime in minutes (1 per run)
    d[f"{prefix}_Runtime_24h_min"] = d[f"{prefix}_Runs_24h"]

    # Cumulative runtime
    d[f"{prefix}_Runtime_Cumulative_min"] = d[f"{prefix}_Runtime_24h_min"].cumsum()

    return d
################################################################################################################################################################

def generate_valve_event_log(df: pd.DataFrame, valve_map: dict) -> pd.DataFrame:
    """
    Adds a 'Valve_Event_Log' column with text describing:
      - Valve open/close events
      - Boolean state transitions
      - Tank fill events with volumes
    """

    df = df.copy()
    event_logs = []

    # --- 1) Discrete valve events (== 1) ---
    for col, message in valve_map.items():
        if col in df.columns:
            triggered = df[col] == 1
            series = pd.Series("", index=df.index)
            series[triggered] = message
            event_logs.append(series)

    # --- 2) Boolean state transitions ---
    bool_map = {
        "is_tank_fill": ("Supply Tank Filling Started", "Supply Tank Filling Ended"),
        "is_pressurising": ("Pressurisation Started", "Pressurisation Ended"),
        "is_fcv_ops": ("FCV Operation Started", "FCV Operation Ended"),
    }

    for col, (start_msg, end_msg) in bool_map.items():
        if col in df.columns:
            start = (df[col].shift(fill_value=False) == False) & (df[col] == True)
            end   = (df[col].shift(fill_value=False) == True) & (df[col] == False)

            start_series = pd.Series("", index=df.index)
            end_series   = pd.Series("", index=df.index)

            start_series[start] = start_msg
            end_series[end]     = end_msg

            event_logs.append(start_series)
            event_logs.append(end_series)

    # --- 3) Tank fill volumes ---
    if "tank_fill_volume" in df.columns:
        fill_series = pd.Series("", index=df.index)
        fills = df.loc[df["tank_fill_volume"] > 0, "tank_fill_volume"]
        for t, vol in fills.items():
            fill_series.loc[t] = f"Supply Tank Filled (+{int(round(vol))} L)"
        event_logs.append(fill_series)

    # --- 4) Combine all events into one log column ---
    if event_logs:
        df["Valve_Event_Log"] = (
            pd.concat(event_logs, axis=1)
            .apply(lambda row: ", ".join([x for x in row if x != ""]), axis=1)
            .replace("", pd.NA)
        )

    return df




######################################################################################################################################################
def run_full_pipeline(
    parquet_path: str,
    columns_to_remove: list,
    valve_columns: list,
    valve5_volume: float,
    valve2_volume: float,
    valve_hp_volume: float,
    pcv_columns: list,
    thresholds: dict,
    pump_events_path: str,
    valve_transition_cols: list,
    drop_slope_data_path: str,
    use_smoothed_tank: bool = True,          
    smoothing_strength: float = 4.8,          
    combine_sensors: str = "mean",
    years_back: int = 8  
) -> pd.DataFrame:
    #Only select the last 8 years of data from the 25 years of data
    df = load_and_clean_otter_data(parquet_path, columns_to_remove, years_back=years_back)
    df = process_valve_data(df, valve_columns)
    df = valve_fluid_usage_calc(df,fluid_per_5lp=valve5_volume,fluid_per_2lp=valve2_volume,fluid_per_hp=valve_hp_volume)
    df = process_fcv(df)
    df = process_pcv(df, pcv_columns)
    df = convert_tank_levels(df)
    df = add_smoothed_supply_level_litres(df,a_L="HPU_SPLY_LEV_L",b_L="HPU_SPLY_LEV_B_L",combine=combine_sensors,strength=3.9,out_col="HPU_SPLY_LEV_L_Smooth")
    df = add_smoothed_return_level_litres(df,a_L="HPU_RET_LEV_L",b_L="HPU-RET_LEV_B_L", combine=combine_sensors,strength=3.9,out_col="HPU_RET_LEV_L_Smooth")
    df = process_umbilical_charges(df,channel_thresholds=thresholds)
    #df = system_fluid_consumption_v2(df,supply_col='HPU_SPLY_LEV_L_Smooth', return_col="HPU_RET_LEV_L_Smooth") # This is the Supply and REturn tanks
    df = system_fluid_consumption(df,supply_col='HPU_SPLY_LEV_L')
    df = add_external_losses(df)
    df = add_external_loss_moving_averages(df)
    df = add_daily_ewm_to_minutely_df(df,loss_col='External_Losses',spans=[1, 7, 30])
    # Use smoothed tank level downstream if requested
    tank_col_to_use = 'HPU_SPLY_LEV_L_Smooth' if use_smoothed_tank else 'HPU_SPLY_LEV_L'
    # Compute those baseline case where there ar eno valve ops and no deamnd on system for 12 hours to get the system fluid use baseline
    df = add_baseline_columns(df,lp_col='SCM1_LP_CONS',hp_col='SCM1_HP_CONS',tank_col=tank_col_to_use,fcv_steps_col='FCV_FullSteps',valve_transition_cols=valve_transition_cols,min_lp=180,min_hp=200,clean_duration=pd.Timedelta('12h'))
    df = add_pressure_deltas(df)
    #Add Pump Runs. Either Load them from the previously ran csv or rerun the function (this takes many hours)
    #df_all_otter = add_pump_run_counts(df_all_otter,lp_col="Cum_LP_A_Run_Count",hp_col="Cum_HP_A_Run_Count")
    df = add_pump_run_cumulatives(df,pump_events_path,pumps=("LP_Pump_A", "HP_Pump_A"))
    #Append previosuly completed gradient drop slopes for LP/HP and Supply Levesl. These had bneen previously run and takes days. Functions to rerun commented out
    df = add_slope_features(df, drop_slope_data_path) # Just got LP and HP
    df = detect_fills_safe(df,level_col="HPU_SPLY_LEV_L",min_event_litres=10.0,step_noise=0.5,max_gap="3min",max_step_litres=150.0,smooth_window="5min")
    df = add_pump_runtime_metrics(df,run_col="Cum_LP_Pump_A_Run_Count",prefix="LP")
    df = add_pump_runtime_metrics(df,run_col="Cum_HP_Pump_A_Run_Count",prefix="HP")
    df = generate_valve_event_log(df, valve_event_map)
    return df
################################################################################################################

def summarise_dataframe(df):
    summary = []

    for col in df.columns:
        dtype = df[col].dtype
        n_missing = df[col].isna().sum()
        n_unique = df[col].nunique(dropna=True)
        example_vals = df[col].dropna().unique()[:5]  # first 5 unique non-null values

        col_summary = {
            'Column': col,
            'DType': dtype,
            'Missing': n_missing,
            'Unique': n_unique,
            'Examples': example_vals
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_summary['Mean'] = df[col].mean()
            col_summary['Std'] = df[col].std()
            col_summary['Min'] = df[col].min()
            col_summary['Max'] = df[col].max()
        else:
            col_summary['Mean'] = col_summary['Std'] = col_summary['Min'] = col_summary['Max'] = None

        summary.append(col_summary)

    return pd.DataFrame(summary)

#########################################################################################################################
#Tank Drop Slopes

def calculate_drop_slopes(df, cols, windows=None, to_per='h'):
    """
    Calculate rolling drop-only slopes for given columns over specified time windows.

    Parameters:
    - df: pandas DataFrame with a datetime index.
    - cols: list of column names to process.
    - windows: dict of label -> window string, e.g., {'1h': '1H'}
    - to_per: 'h' for L/h output, 's' for L/s output.

    Returns:
    - Modified DataFrame with new slope columns added.
    """
    # Ensure datetime index, sorted, de-duplicated
    d = df.copy()
    d.index = pd.to_datetime(d.index, errors='coerce')
    d = d[~d.index.isna()].sort_index()
    if d.index.has_duplicates:
        d = d[~d.index.duplicated(keep='last')]

    # Default windows if none provided
    if windows is None:
        windows = {'1h': '1H', '2h': '2H', '8h': '8H'}

    # Internal function: slope per second, stable
    def _slope_seconds_safe(x: pd.Series):
        # times in seconds from window start (improves conditioning)
        t = (x.index.view('int64') // 10**9).astype('float64')
        y = x.values.astype('float64')
        mask = np.isfinite(y)
        t = t[mask]; y = y[mask]
        if t.size < 2 or np.nanstd(y) == 0:
            return np.nan
        t0 = t[0]
        t = t - t0
        A = np.vstack([t, np.ones_like(t)]).T
        try:
            a, _b = np.linalg.lstsq(A, y, rcond=None)[0]
            return float(a)  # L per second
        except np.linalg.LinAlgError:
            return np.nan

    # Rolling slope wrapper
    def rolling_slope(series: pd.Series, window: str, to_per='h'):
        slopes_per_s = series.rolling(window, min_periods=2).apply(_slope_seconds_safe, raw=False)
        return slopes_per_s * (3600 if to_per == 'h' else 1)

    # Compute slopes
    for col in cols:
        if col not in d.columns:
            # keep going, but warn in stdout
            print(f"⚠️  calculate_drop_slopes: column '{col}' not found; skipping.")
            continue

        drop_only = pd.to_numeric(d[col], errors='coerce').diff().clip(upper=0).abs()
        for label, win in windows.items():
            new_col = f"{col}_DropSlope_{label}_Lph" if to_per == 'h' else f"{col}_DropSlope_{label}_Lps"
            if new_col not in d.columns:
                d[new_col] = rolling_slope(drop_only, win, to_per)

    return d

################################################################################################################################
# Pressure drop slopes
from typing import Optional, Dict, List, Callable
import numpy as np
import pandas as pd

# ---------- core kernels (Numba-friendly raw=True uses ndarray) ----------

def _pressure_slope_nb(a: np.ndarray, to_per: str = "h") -> float:
    """
    Linear slope of pressure over window 'a' using centered time in minutes.
    Returns slope in (input units) per 'to_per' where to_per in {'h','min','s'}.
    """
    if a.size < 2 or np.isnan(a).any():
        return np.nan

    # t in minutes, centered to improve conditioning
    t = np.arange(a.size, dtype=np.float64)
    t -= t.mean()
    y = a.astype(np.float64)
    y -= y.mean()

    denom = (t * t).sum()
    if denom == 0.0:
        return 0.0

    slope_per_min = (t * y).sum() / denom  # units per minute

    if to_per == "h":
        return slope_per_min * 60.0
    elif to_per == "min":
        return slope_per_min
    elif to_per == "s":
        return slope_per_min / 60.0
    else:
        # default to per-hour if unknown
        return slope_per_min * 60.0

def _pressure_slope_general(win: pd.Series, to_per: str = "h") -> float:
    """
    Time-aware slope using actual timestamps (handles gaps/irregular sampling).
    Uses least squares on seconds-from-window-start, robust to NaNs.
    """
    idx_s = win.index.view("i8") / 1e9  # seconds
    y = win.values.astype("float64")
    good = np.isfinite(y)
    if good.sum() < 2:
        return np.nan

    t = idx_s[good]
    y = y[good]
    t -= t[0]  # seconds from window start
    A = np.vstack([t, np.ones_like(t)]).T
    try:
        a, _b = np.linalg.lstsq(A, y, rcond=None)[0]  # a in units per second
        if to_per == "h":
            return float(a * 3600.0)
        elif to_per == "min":
            return float(a * 60.0)
        elif to_per == "s":
            return float(a)
        else:
            return float(a * 3600.0)
    except np.linalg.LinAlgError:
        return np.nan

# ---------- public function ----------

def calculate_pressure_slopes_fast(
    df: pd.DataFrame,
    cols: List[str],
    windows: Optional[Dict[str, str]] = None,  # e.g., {'1h':'60T','6h':'360T'}
    to_per: str = "h",                          # 'h' | 'min' | 's'
    prefer_numba: bool = True,
    # progress options (disables numba if used)
    progress: Optional[str] = None,             # None | "print" | "tqdm"
    progress_every: int = 1_000,
    progress_prefix: str = "pressure",
    progress_fn: Optional[Callable[[pd.Timestamp, int, int], None]] = None,
) -> pd.DataFrame:
    """
    Rolling linear slope of pressure signals.
    Adds columns like: <col>_PressureSlope_<label>_perh  (or permin/pers).
    Uses Pandas+Numba fast path for fixed-size minute windows with raw=True.
    Falls back to Python engine for time-based windows or when progress is requested.
    """
    d = df.copy()
    d.index = pd.to_datetime(d.index, errors="coerce")
    d = d[~d.index.isna()].sort_index()

    if windows is None:
        windows = {"1h": "60T"}  # default 1 hour in minutes

    # minutely check to enable the Numba path for fixed-size windows
    is_minutely = _infer_minutely(d.index)

    # check Numba availability only if we *could* use it
    have_numba = False
    if prefer_numba and progress is None and is_minutely:
        try:
            import numba  # noqa: F401
            have_numba = True
        except Exception:
            have_numba = False

    # optional tqdm
    use_tqdm = (progress == "tqdm")
    if use_tqdm:
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None
            use_tqdm = False
            if progress == "tqdm":
                print("⚠️ tqdm not available; falling back to print progress.")

    per_suffix = {"h": "perh", "min": "permin", "s": "pers"}.get(to_per, "perh")

    for col in cols:
        if col not in d.columns:
            print(f"⚠️ calculate_pressure_slopes_fast: '{col}' not in df; skipping.")
            continue

        s = pd.to_numeric(d[col], errors="coerce")

        for label, win in windows.items():
            out_col = f"{col}_PressureSlope_{label}_{per_suffix}"

            if is_minutely:
                # ---------- fixed-size windows => numba-capable branch ----------
                W = _window_to_int_minutes(win)  # minutes -> int
                roll = s.rolling(W, min_periods=2)

                if progress is None and have_numba:
                    # FAST PATH: Pandas rolling with Numba engine
                    print(f"[pressure-slope] using numba engine on {len(s)} rows; window={W} min for col={col}/{label}")
                    d[out_col] = roll.apply(
                        lambda a: _pressure_slope_nb(a, to_per=to_per),
                        raw=True,
                        engine="numba",
                        engine_kwargs={"nopython": True, "nogil": True},
                    )
                elif progress is None:
                    # Python engine without progress
                    d[out_col] = roll.apply(lambda a: _pressure_slope_nb(a, to_per=to_per), raw=True)
                else:
                    # Progress requested -> Python engine with callbacks
                    idx_list = s.index
                    n_calls = len(s)
                    i_counter = {"i": -1}
                    if use_tqdm:
                        pbar = tqdm(total=n_calls, desc=f"{progress_prefix}:{col}:{label}", leave=False)

                    def _wrapped_raw(a: np.ndarray) -> float:
                        i_counter["i"] += 1
                        i = i_counter["i"]
                        if use_tqdm:
                            pbar.update(1)
                        elif progress == "print" and (i % progress_every == 0 or i == n_calls - 1):
                            print(f"[{progress_prefix}] {col} {label}: {i+1}/{n_calls} @ {idx_list[min(i, len(idx_list)-1)]}")
                        if callable(progress_fn):
                            try:
                                progress_fn(idx_list[min(i, len(idx_list)-1)], i+1, n_calls)
                            except Exception:
                                pass
                        return _pressure_slope_nb(a, to_per=to_per)

                    try:
                        d[out_col] = roll.apply(_wrapped_raw, raw=True)
                    finally:
                        if use_tqdm:
                            pbar.close()

            else:
                # ---------- time-based windows => Python engine with Series ----------
                roll = s.rolling(str(win), min_periods=2)

                if progress is None:
                    d[out_col] = roll.apply(lambda w: _pressure_slope_general(w, to_per=to_per), raw=False)
                else:
                    idx_list = s.index
                    n_calls = len(s)
                    i_counter = {"i": -1}
                    if use_tqdm:
                        pbar = tqdm(total=n_calls, desc=f"{progress_prefix}:{col}:{label}", leave=False)

                    def _wrapped_series(w: pd.Series) -> float:
                        i_counter["i"] += 1
                        i = i_counter["i"]
                        if use_tqdm:
                            pbar.update(1)
                        elif progress == "print" and (i % progress_every == 0 or i == n_calls - 1):
                            print(f"[{progress_prefix}] {col} {label}: {i+1}/{n_calls} @ {idx_list[min(i, len(idx_list)-1)]}")
                        if callable(progress_fn):
                            try:
                                progress_fn(idx_list[min(i, len(idx_list)-1)], i+1, n_calls)
                            except Exception:
                                pass
                        return _pressure_slope_general(w, to_per=to_per)

                    try:
                        d[out_col] = roll.apply(_wrapped_series, raw=False)
                    finally:
                        if use_tqdm:
                            pbar.close()

    return d
#############################################################################################
import numpy as np
import pandas as pd
import numba

def _infer_minutely(dti: pd.DatetimeIndex) -> bool:
    if not isinstance(dti, pd.DatetimeIndex) or len(dti) < 5: 
        return False
    diffs = dti.view("i8")[1:] - dti.view("i8")[:-1]
    sixty = 60 * 1_000_000_000
    return (np.abs(diffs - sixty) < 5_000_000).mean() > 0.999

def _window_to_int_minutes(win: str) -> int:
    win = str(win).upper()
    if win.endswith("H"): return int(float(win[:-1]) * 60)
    if win.endswith("T") or win.endswith("MIN"): return int(float(win.rstrip("TMIN")))
    if win.endswith("D"): return int(float(win[:-1]) * 1440)
    raise ValueError(f"Unsupported window '{win}'")

def _drop_slope_lph_nb(a):
    # a is drop-only series already (non-positive diffs made positive); ndarray expected when raw=True
    if np.isnan(a).any() or a.size < 2:
        return np.nan
    t = np.arange(a.size, dtype=np.float64)
    t -= t.mean()
    y = a.astype(np.float64)
    y -= np.mean(y)
    denom = (t * t).sum()
    if denom == 0.0:
        return 0.0
    m_per_min = (t * y).sum() / denom
    return m_per_min * 60.0  # L per hour

def _drop_slope_lph_general(win: pd.Series) -> float:
    idx = win.index.view("i8") / 1e9
    y = win.values.astype("float64")
    good = np.isfinite(y)
    if good.sum() < 2:
        return np.nan
    t = idx[good]
    y = y[good]
    t -= t[0]
    A = np.vstack([t, np.ones_like(t)]).T
    try:
        a, _b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(a * 3600.0)
    except np.linalg.LinAlgError:
        return np.nan

def calculate_drop_slopes_fast(
    df: pd.DataFrame,
    cols: list,
    windows: dict = None,     # {'1h':'1H','6h':'6H'}
    to_per: str = 'h',        # only 'h' supported here
    prefer_numba: bool = True,
    # ---- new progress options ----
    progress: str | None = None,   # None | "print" | "tqdm"
    progress_every: int = 500,     # print/update every N datapoints
    progress_prefix: str = "calc",
    progress_fn=None               # optional callback(idx, i, n)
) -> pd.DataFrame:
    """
    If progress is requested, falls back to the Python engine for that rolling apply
    so we can emit per-window progress. Numba path cannot report progress.
    """
    d = df.copy()
    d.index = pd.to_datetime(d.index, errors="coerce")
    d = d[~d.index.isna()].sort_index()

    if windows is None:
        windows = {'1h':'1H','6h':'6H'}

    is_minutely = _infer_minutely(d.index)

    # --- Numba availability check (import is enough for Pandas engine="numba") ---
    have_numba = False
    if prefer_numba and progress is None and is_minutely:
        try:
            import numba  # noqa: F401
            have_numba = True
        except Exception:
            have_numba = False

    # optional tqdm setup
    use_tqdm = (progress == "tqdm")
    if use_tqdm:
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None
            use_tqdm = False
            if progress == "tqdm":
                print("⚠️ tqdm not available; falling back to print progress.")

    for col in cols:
        if col not in d.columns:
            print(f"⚠️ calculate_drop_slopes_fast: '{col}' not in df; skipping.")
            continue

        # drop-only (convert non-positive diffs to positive magnitudes)
        s = pd.to_numeric(d[col], errors="coerce")
        drop_only = s.diff().clip(upper=0).abs()

        for label, win in windows.items():
            out_col = f"{col}_DropSlope_{label}_Lph"

            if is_minutely:
                # ---- fixed-size window (int minutes) -> can use numba engine with raw=True ----
                W = _window_to_int_minutes(win)
                roll = drop_only.rolling(W, min_periods=2)

                if progress is None and have_numba:
                    # Fast path: Pandas+Numba JIT (Numba 0.58.1 friendly)
                    # NOTE: engine_kwargs must include nopython=True; nogil is a good default.
                    print(f"[drop-slopes] using numba engine on {len(drop_only)} rows; window={W} min for col={col}/{label}")
                    d[out_col] = roll.apply(
                        _drop_slope_lph_nb,
                        raw=True,
                        engine="numba",
                        engine_kwargs={"nopython": True, "nogil": True}
                    )
                elif progress is None:
                    # Python engine fallback without progress
                    d[out_col] = roll.apply(_drop_slope_lph_nb, raw=True)
                else:
                    # Progress requested -> wrap and run with Python engine (cannot callback from numba)
                    idx_list = drop_only.index
                    n_calls = len(drop_only)
                    i_counter = {"i": -1}

                    if use_tqdm:
                        pbar = tqdm(total=n_calls, desc=f"{progress_prefix}:{col}:{label}", leave=False)

                    def _wrapped_raw(a):
                        i_counter["i"] += 1
                        i = i_counter["i"]
                        if use_tqdm:
                            pbar.update(1)
                        elif progress == "print" and (i % progress_every == 0 or i == n_calls - 1):
                            print(f"[{progress_prefix}] {col} {label}: {i+1}/{n_calls} @ {idx_list[min(i, len(idx_list)-1)]}")
                        if callable(progress_fn):
                            try:
                                progress_fn(idx_list[min(i, len(idx_list)-1)], i+1, n_calls)
                            except Exception:
                                pass
                        return _drop_slope_lph_nb(a)

                    if have_numba:
                        print(f"ℹ️ Progress requested for {col}/{label}; running without numba to report updates.")
                    try:
                        d[out_col] = roll.apply(_wrapped_raw, raw=True)
                    finally:
                        if use_tqdm:
                            pbar.close()

            else:
                # ---- time-based window -> requires Series (raw=False). Keep Python engine. ----
                roll = drop_only.rolling(str(win), min_periods=2)

                if progress is None:
                    d[out_col] = roll.apply(_drop_slope_lph_general, raw=False)
                else:
                    idx_list = drop_only.index
                    n_calls = len(drop_only)
                    i_counter = {"i": -1}

                    if use_tqdm:
                        pbar = tqdm(total=n_calls, desc=f"{progress_prefix}:{col}:{label}", leave=False)

                    def _wrapped_series(win_series: pd.Series):
                        i_counter["i"] += 1
                        i = i_counter["i"]
                        if use_tqdm:
                            pbar.update(1)
                        elif progress == "print" and (i % progress_every == 0 or i == n_calls - 1):
                            print(f"[{progress_prefix}] {col} {label}: {i+1}/{n_calls} @ {idx_list[min(i, len(idx_list)-1)]}")
                        if callable(progress_fn):
                            try:
                                progress_fn(idx_list[min(i, len(idx_list)-1)], i+1, n_calls)
                            except Exception:
                                pass
                        return _drop_slope_lph_general(win_series)

                    try:
                        d[out_col] = roll.apply(_wrapped_series, raw=False)
                    finally:
                        if use_tqdm:
                            pbar.close()

    return d


#############################################################
def add_pump_run_counts(
    df,
    lp_col="Cum_LP_A_Run_Count",
    hp_col="Cum_HP_A_Run_Count",
    lp_prefix="LP",
    hp_prefix="HP"
):
    """
    From cumulative run counts, compute run-event indicators and rolling sums
    over 2 hours and 24 hours for LP and HP.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DateTimeIndex and cumulative count columns.
    lp_col : str
        Column name for LP cumulative run count.
    hp_col : str
        Column name for HP cumulative run count.
    lp_prefix : str
        Prefix for new LP columns.
    hp_prefix : str
        Prefix for new HP columns.

    Returns
    -------
    df : pd.DataFrame
        Original DataFrame with new columns added:
        - <prefix>_Runs_2h
        - <prefix>_Runs_24h
    """

    d = df.copy()

    # --- 1) Run events (differences in cumulative counts) ---
    d[f"{lp_prefix}_Run_Event"] = (
        d[lp_col].diff().clip(lower=0).fillna(0)
    )
    d[f"{hp_prefix}_Run_Event"] = (
        d[hp_col].diff().clip(lower=0).fillna(0)
    )

    # --- 2) Rolling sums (2h and 24h windows) ---
    for prefix in [lp_prefix, hp_prefix]:
        d[f"{prefix}_Runs_2h"] = d[f"{prefix}_Run_Event"].rolling("2H", min_periods=1).sum()
        d[f"{prefix}_Runs_24h"] = d[f"{prefix}_Run_Event"].rolling("24H", min_periods=1).sum()

    # --- 3) Drop intermediate event columns ---
    d = d.drop(columns=[f"{lp_prefix}_Run_Event", f"{hp_prefix}_Run_Event"])

    return d
###########################################################################
################################################################################################################################################################
import pandas as pd

def clean_dataframe_and_split_valves(df: pd.DataFrame):
    # --- 0) Work on a copy
    df = df.copy()

    # --- 1) Drop unnecessary columns (your updated list) ---
    cols_to_drop = {
        # HPU level duplicates / unused
        'HPU_SPLY_LEV_B','HPU-RET_LEV_B','HPU_SPLY_LEV','HPU_RET_LEV',
        'HPU_LP_B_SPLY','HPU_HP_B_SPLY','HPU_SPLY_LEV_B_L','HPU-RET_LEV_B_L',

        # SCM2–5 extra channels (keep SCM1 + SCM2 COVs you want split out)
        'SCM2_LP_CONS','SCM2_HP_CONS','SCM2_RET',
        'SCM3_LP1_COV','SCM3_HP_COV','SCM4_LP1_COV','SCM4_HP_COV',
        'SCM5_LP1_COV','SCM5_HP_COV',
        'SCM3_LP_CONS','SCM3_HP_CONS','SCM3_RET',
        'SCM4_LP_CONS','SCM4_HP_CONS','SCM4_RET',
        'SCM5_LP_CONS','SCM5_HP_CONS','SCM5_RET',

        # PCV raw channels
        'P1_PCV','P2_PCV','P3_PCV','I1_PCV','I2_PCV',

        # Well valve transitions
        'P1_PMV_OpenToClosed','P1_PMV_ClosedToOpen','P1_PWV_OpenToClosed','P1_PWV_ClosedToOpen',
        'P1_AMV_OpenToClosed','P1_AMV_ClosedToOpen','P1_SCSSV_OpenToClosed','P1_SCSSV_ClosedToOpen',
        'P1_PDV_OpenToClosed','P1_PDV_ClosedToOpen','P1_TDV_OpenToClosed','P1_TDV_ClosedToOpen',
        'P2_PMV_OpenToClosed','P2_PMV_ClosedToOpen','P2_PWV_OpenToClosed','P2_PWV_ClosedToOpen',
        'P2_AMV_OpenToClosed','P2_AMV_ClosedToOpen','P2_SCSSV_OpenToClosed','P2_SCSSV_ClosedToOpen',
        'P2_PDV_OpenToClosed','P2_PDV_ClosedToOpen','P2_TDV_OpenToClosed','P2_TDV_ClosedToOpen',
        'P3_PMV_OpenToClosed','P3_PMV_ClosedToOpen','P3_PWV_OpenToClosed','P3_PWV_ClosedToOpen',
        'P3_AMV_OpenToClosed','P3_AMV_ClosedToOpen','P3_SCSSV_OpenToClosed','P3_SCSSV_ClosedToOpen',
        'P3_PDV_OpenToClosed','P3_PDV_ClosedToOpen','P3_TDV_OpenToClosed','P3_TDV_ClosedToOpen',
        'I1_PMV_OpenToClosed','I1_PMV_ClosedToOpen','I1_PWV_OpenToClosed','I1_PWV_ClosedToOpen',
        'I1_AMV_OpenToClosed','I1_AMV_ClosedToOpen',
        'I2_PMV_OpenToClosed','I2_PMV_ClosedToOpen','I2_PWV_OpenToClosed','I2_PWV_ClosedToOpen',
        'I2_AMV_OpenToClosed','I2_AMV_ClosedToOpen',
        'MPMV_Inlet_OpenToClosed','MPMV_Inlet_ClosedToOpen',
        'Man_CI_OpenToClosed','Man_CI_ClosedToOpen',
        'SCM1_LP1_COV_OpenToClosed','SCM1_LP1_COV_ClosedToOpen',
        'SCM1_HP_COV_OpenToClosed','SCM1_HP_COV_ClosedToOpen',

        # Valve/actuation totals & steps
        #'Valve_Operation_Fluid','Cumulative_Valve_Operation_Fluid',
        #'FCV_CPI_FullSteps','FCV_FullSteps',
        'PCV_FullSteps','PCV_Fluid_Usage',
        'Cumulative_FCV_Fluid_Usage','Cumulative_PCV_Fluid_Usage',

        # Umbilical (keep instant, drop cumulative) + legacy 30d EWM
        'cum_umbilical_charge_volume','EWM_30d',

        # Pump cumulatives
        'Cum_LP_Pump_A_Run_Count','Cum_HP_Pump_A_Run_Count',
        'Cum_LP_Pump_A_Run_Dur','Cum_HP_Pump_A_Run_Dur',

        # Supply consumption (rate & cumulative)
        'Supply_Consumption_Rate_L_per_h','Cumulative_Supply_Consumption_Excl_Fills',

        # Baseline & slopes you wanted to drop
        #'baseline_drop_L','baseline_drift_L_per_day',
        'Slope_1H_Lph','Slope_7D_Lph','External_Loss_EWM_30d',

        # Misc counters / intermediate flags
        'P3_PDV_num',
        'FCV_Operation','combined_fill_events',
        'is_HP_pump_normal','is_LP_pump_normal',

        # Run-count / rate aggregates (24h + some 2h)
        'LP_Runs_2h','HP_Runs_24h','HP_Runs_2h',
        'HP_pump_rate_24h','LP_pump_rate_24h',
    }
    df.drop(columns=list(set(df.columns) & cols_to_drop), inplace=True)

    # --- 2) Split out valve position/state columns (keep FCV_CPI in main df) ---
    valve_position_cols = [
        # P3 well
        'P3_PMV','P3_PWV','P3_AMV','P3_SCSSV','P3_PDV','P3_TDV',
        # P1 well
        'P1_PMV','P1_PWV','P1_AMV','P1_SCSSV','P1_PDV','P1_TDV',
        # P2 well
        'P2_PMV','P2_PWV','P2_AMV','P2_SCSSV','P2_PDV','P2_TDV',
        # Injectors
        'I1_PMV','I1_PWV','I1_AMV','I2_PMV','I2_PWV','I2_AMV',
        # Manifold / COVs
        'MPMV_Inlet','Man_CI','SCM1_LP1_COV','SCM1_HP_COV','SCM2_LP1_COV','SCM2_HP_COV',
    ]
    valve_present = [c for c in valve_position_cols if c in df.columns]
    otter_valve_positions_df = df[valve_present].copy()
    df.drop(columns=valve_present, inplace=True)

    # --- 3) Move key HPU columns to front (with Smooth right after RET) ---
    lead = []
    if 'HPU_SPLY_LEV_L' in df.columns: lead.append('HPU_SPLY_LEV_L')
    if 'HPU_RET_LEV_L'  in df.columns: lead.append('HPU_RET_LEV_L')
    if 'HPU_SPLY_LEV_L_Smooth' in df.columns: lead.append('HPU_SPLY_LEV_L_Smooth')
    if 'HPU_RET_LEV_L_Smooth'  in df.columns: lead.append('HPU_RET_LEV_L_Smooth')
    df = df[lead + [c for c in df.columns if c not in lead]]

    # --- 4) Rename slopes + MA/EWM ---
    df.rename(columns={
        'Slope_2H_Lph':'Supply_Slope_2H_Lph',
        'Slope_12H_Lph':'Supply_Slope_12H_Lph',
        'Slope_24H_Lph':'Supply_Slope_24H_Lph',
        'MA_2h':'External_Loss_MA_2h',
        'MA_12h':'External_Loss_MA_12h',
        'MA_24h':'External_Loss_MA_24h',
        'MA_168h':'External_Loss_MA_168h',
        'EWM_1d':'External_Loss_EWM_1d',
        'EWM_7d':'External_Loss_EWM_7d',
    }, inplace=True)

    # --- 5) Ensure numeric on FCV fields you keep (FCV_CPI, FCV_Fluid_Usage) ---
    for col in ('FCV_CPI','FCV_Fluid_Usage'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- 6) Encode valve states in the split DataFrame (case-insensitive) ---
    if not otter_valve_positions_df.empty:
        state_map = {"OPEN":1, "CLOSED":0, "FAULT":-1, "NO DATA":-1, "TIMEOUT":-1, "UNKNOWN":-1}
        for col in otter_valve_positions_df.columns:
            s = otter_valve_positions_df[col].astype(str).str.upper()
            otter_valve_positions_df[col] = s.map(state_map).fillna(-1).astype('int32')

        # --- 7) Your override: FAULT (-1) -> OPEN (1) for specific valves ---
        for col in ('P3_PDV','I2_AMV','P3_PWV','P3_AMV'):
            if col in otter_valve_positions_df.columns:
                otter_valve_positions_df[col] = otter_valve_positions_df[col].replace(-1, 1)

    # --- 8) Reorder pump metrics before is_consumption_low (in main df) ---
    if 'is_consumption_low' in df.columns:
        for c in ('HP_pump_rate_2h','HP_pump_state','LP_pump_rate_2h','LP_pump_state'):
            if c in df.columns:
                df.insert(df.columns.get_loc('is_consumption_low'), c, df.pop(c))
    
    losses_col = "is_losses_high"
    cons_low_col = "is_consumption_low"  

    missing = [c for c in [losses_col, cons_low_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    df["is_steady_state"] = (df[losses_col] == 0).astype(int)
    df["is_steady_state_strict"] = (df[cons_low_col] == 1).astype(int)
   
    # Quick sanity check
    print(f"steady_state = {100*df['is_steady_state'].mean():.1f}%")
    print(f"steady_state_strict = {100*df['is_steady_state_strict'].mean():.1f}%")
    
    
    # Keep both with same DatetimeIndex (assumes df had one)
    otter_valve_positions_df = otter_valve_positions_df.reindex(df.index)

    return df, otter_valve_positions_df
########################################################################################################################################
def clean_dataframe(df):
    # 1. Drop unnecessary columns
    cols_to_drop = {
        # HPU level duplicates / unused
        'HPU_SPLY_LEV_B','HPU-RET_LEV_B','HPU_SPLY_LEV','HPU_RET_LEV',
        'HPU_LP_B_SPLY','HPU_HP_B_SPLY','HPU_SPLY_LEV_B_L','HPU-RET_LEV_B_L',

        # SCM2–5 extra channels (keep SCM1 + SCM2 COVs you want split out)
        'SCM2_LP_CONS','SCM2_HP_CONS','SCM2_RET',
        'SCM3_LP1_COV','SCM3_HP_COV','SCM4_LP1_COV','SCM4_HP_COV',
        'SCM5_LP1_COV','SCM5_HP_COV',
        'SCM3_LP_CONS','SCM3_HP_CONS','SCM3_RET',
        'SCM4_LP_CONS','SCM4_HP_CONS','SCM4_RET',
        'SCM5_LP_CONS','SCM5_HP_CONS','SCM5_RET',

        # PCV raw channels
        'P1_PCV','P2_PCV','P3_PCV','I1_PCV','I2_PCV',

        # Well valve transitions
        'P1_PMV_OpenToClosed','P1_PMV_ClosedToOpen','P1_PWV_OpenToClosed','P1_PWV_ClosedToOpen',
        'P1_AMV_OpenToClosed','P1_AMV_ClosedToOpen','P1_SCSSV_OpenToClosed','P1_SCSSV_ClosedToOpen',
        'P1_PDV_OpenToClosed','P1_PDV_ClosedToOpen','P1_TDV_OpenToClosed','P1_TDV_ClosedToOpen',
        'P2_PMV_OpenToClosed','P2_PMV_ClosedToOpen','P2_PWV_OpenToClosed','P2_PWV_ClosedToOpen',
        'P2_AMV_OpenToClosed','P2_AMV_ClosedToOpen','P2_SCSSV_OpenToClosed','P2_SCSSV_ClosedToOpen',
        'P2_PDV_OpenToClosed','P2_PDV_ClosedToOpen','P2_TDV_OpenToClosed','P2_TDV_ClosedToOpen',
        'P3_PMV_OpenToClosed','P3_PMV_ClosedToOpen','P3_PWV_OpenToClosed','P3_PWV_ClosedToOpen',
        'P3_AMV_OpenToClosed','P3_AMV_ClosedToOpen','P3_SCSSV_OpenToClosed','P3_SCSSV_ClosedToOpen',
        'P3_PDV_OpenToClosed','P3_PDV_ClosedToOpen','P3_TDV_OpenToClosed','P3_TDV_ClosedToOpen',
        'I1_PMV_OpenToClosed','I1_PMV_ClosedToOpen','I1_PWV_OpenToClosed','I1_PWV_ClosedToOpen',
        'I1_AMV_OpenToClosed','I1_AMV_ClosedToOpen',
        'I2_PMV_OpenToClosed','I2_PMV_ClosedToOpen','I2_PWV_OpenToClosed','I2_PWV_ClosedToOpen',
        'I2_AMV_OpenToClosed','I2_AMV_ClosedToOpen',
        'MPMV_Inlet_OpenToClosed','MPMV_Inlet_ClosedToOpen',
        'Man_CI_OpenToClosed','Man_CI_ClosedToOpen',
        'SCM1_LP1_COV_OpenToClosed','SCM1_LP1_COV_ClosedToOpen',
        'SCM1_HP_COV_OpenToClosed','SCM1_HP_COV_ClosedToOpen',

        # Valve/actuation totals & steps
        'Cumulative_Valve_Operation_Fluid',
        'FCV_CPI_FullSteps','FCV_FullSteps',
        'PCV_FullSteps',
        'Cumulative_FCV_Fluid_Usage','Cumulative_PCV_Fluid_Usage',

        # Umbilical (keep instant, drop cumulative) + legacy 30d EWM
        'cum_umbilical_charge_volume','EWM_30d',

        # Pump cumulatives
      
        'Cum_LP_Pump_A_Run_Dur','Cum_HP_Pump_A_Run_Dur',

        # Supply consumption (rate & cumulative)
        'Supply_Consumption_Rate_L_per_h','Cumulative_Supply_Consumption_Excl_Fills',

        # Baseline & slopes you wanted to drop
        #'baseline_drop_L','baseline_drift_L_per_day',
        'Slope_1H_Lph','Slope_7D_Lph','External_Loss_EWM_30d',

        # Misc counters / intermediate flags
        'P3_PDV_num',
        'FCV_Operation','combined_fill_events',
        'is_HP_pump_normal','is_LP_pump_normal',

        # Run-count / rate aggregates (24h + some 2h)
        'LP_Runs_2h','HP_Runs_24h','HP_Runs_2h',
        'HP_pump_rate_24h','LP_pump_rate_24h',
    }
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # 2. Move key columns to front
    lead_cols = []
    if 'HPU_SPLY_LEV_L' in df.columns:
        lead_cols.append('HPU_SPLY_LEV_L')
    if 'HPU_RET_LEV_L' in df.columns:
        lead_cols.append('HPU_RET_LEV_L')
    if 'HPU_SPLY_LEV_L_Smooth' in df.columns:  # <--- adjusted ordering for HPU smooth cols
        lead_cols.append('HPU_SPLY_LEV_L_Smooth')
    if 'HPU_RET_LEV_L_Smooth' in df.columns:
        lead_cols.append('HPU_RET_LEV_L_Smooth')

    df = df[lead_cols + [c for c in df.columns if c not in lead_cols]]

    # 3. Rename slope and MA/EWM columns
    slope_rename_map = {
        'Slope_2H_Lph': 'Supply_Slope_2H_Lph',
        'Slope_12H_Lph': 'Supply_Slope_12H_Lph',
        'Slope_24H_Lph': 'Supply_Slope_24H_Lph',
    }
    external_rename_map = {
        'MA_2h': 'External_Loss_MA_2h',
        'MA_12h': 'External_Loss_MA_12h',
        'MA_24h': 'External_Loss_MA_24h',
        'MA_168h': 'External_Loss_MA_168h',
        'EWM_1d': 'External_Loss_EWM_1d',
        'EWM_7d': 'External_Loss_EWM_7d',
    }
    df = df.rename(columns={**slope_rename_map, **external_rename_map})

    # 5. Clean FCV columns
    for col in ['FCV_CPI', 'FCV_Fluid_Usage']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 6. Encode valve states
    valve_state_map = {
        "OPEN": 1,
        "CLOSED": 0,
        "FAULT": -1,
        "No Data": -1,
        "TIMEOUT": -1,
        "UNKNOWN": -1,
    }
    valve_columns = [
        'P1_PMV', 'P1_PWV', 'P1_AMV', 'P1_SCSSV', 'P1_PDV', 'P1_TDV',
        'P2_PMV', 'P2_PWV', 'P2_AMV', 'P2_SCSSV', 'P2_PDV', 'P2_TDV',
        'P3_PMV', 'P3_PWV', 'P3_AMV', 'P3_SCSSV', 'P3_PDV', 'P3_TDV',
        'I1_PMV', 'I1_PWV', 'I1_AMV',
        'I2_PMV', 'I2_PWV', 'I2_AMV',
        'MPMV_Inlet', 'Man_CI', 'SCM1_LP1_COV', 'SCM1_HP_COV'
    ]
    for col in valve_columns:
        if col in df.columns:
            df[col] = df[col].map(valve_state_map).fillna(-1).astype(int)

    # 7. Special override: treat FAULT as OPEN
    override_fault_open = ['P3_PDV', 'I2_AMV', 'P3_PWV', 'P3_AMV']
    for col in override_fault_open:
        if col in df.columns:
            df[col] = df[col].replace(-1, 1)

    pump_cols = ["HP_pump_rate_2h","HP_pump_state","LP_pump_rate_2h","LP_pump_state"]
    if "is_consumption_low" in df.columns:
        for c in pump_cols:
            if c in df.columns:
                df.insert(df.columns.get_loc("is_consumption_low"), c, df.pop(c))

    return df
