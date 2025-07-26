
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
fcv_step_volume=0.7
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

#For umb charge event sid
thresholds = {
    'SCM1_LP_CONS': (40, 190),    # LPA
    'SCM1_HP_CONS': (40, 450),    # HPA
    'HPU_LPB_OUT': (30, 180),     # LPB pressure indicator
    'HPU_HPB_OUT': (30, 440),     # HPB pressure indicator
    'HPU_LPA_OUT': (20, 180),     # LPB pressure indicator
    'HPU_HPA_OUT': (20, 440),     # HPB pressure indicator
}
######################################################################################################################################################################
def load_and_clean_otter_data(parquet_path: str,columns_to_remove: list) -> pd.DataFrame:
    """
    Reads the Otter dataset from a Parquet file, converts & sorts the timestamp,
    renames misnamed columns, drops unused columns, and returns the cleaned DataFrame.
    """
    # 1) Read
    df_all_otter = pd.read_parquet(parquet_path)

    # 2) Timestamp → datetime index
    df_all_otter['Timestamp'] = pd.to_datetime(
        df_all_otter['Timestamp'], errors='coerce'
    )
    df_all_otter.set_index('Timestamp', inplace=True)
    df_all_otter.sort_index(inplace=True)

    # 3) Rename misnamed column
    df_all_otter.rename(
        columns={'HPU-RET_LEV': 'HPU_RET_LEV'},
        inplace=True
    )

    # only drop the ones that actually exist
    existing = [c for c in columns_to_remove if c in df_all_otter.columns]
    df_all_otter.drop(columns=existing, inplace=True)

    # 5) Report & return
    print(f"Loaded {df_all_otter.shape[0]} rows × {df_all_otter.shape[1]} cols")
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
    fluid_per_step: float = None
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
    supply_pct_col: str = 'HPU_SPLY_LEV',
    return_pct_col: str = 'HPU_RET_LEV',
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
    
    # convert %
    df[f"{supply_pct_col}_L"] = df[supply_pct_col] * (supply_volume / 100.0)
    df[f"{return_pct_col}_L"] = df[return_pct_col] * (return_volume  / 100.0)
    
    print("✅ Tank levels converted to litres:", f"{supply_pct_col}_L", "&", f"{return_pct_col}_L")
    return df

################################################################################################################################################################
def detect_fill_events(
    df: pd.DataFrame,
    main_period: int = 120,
    main_threshold: float = 30.0,
    backup_period: int = 90,
    backup_threshold: float = 25.0,
    level_col: str = 'HPU_SPLY_LEV_L'
) -> pd.DataFrame:
    """
    Detects supply‐tank fill events and computes corrected per‐fill and cumulative volumes.
    Adds columns:
      - combined_fill_events
      - block_id
      - Fluid_Added_Corrected
      - Cumulative_Fluid_Added_Corrected
    Prints total corrected fill volume across all events.
    """
    df = df.copy()
    
    # 1) Ensure datetime index & sorted
    df.index = pd.to_datetime(df.index, errors='coerce')
    df.sort_index(inplace=True)
    
    # 2) Detect raw fill events
    diff_main   = df[level_col].diff(periods=main_period)
    diff_backup = df[level_col].diff(periods=backup_period)
    fill_main   = diff_main.where(diff_main > main_threshold, 0)
    fill_backup = diff_backup.where(diff_backup > backup_threshold, 0)
    df['combined_fill_events'] = pd.concat([fill_main, fill_backup], axis=1).max(axis=1)
    
    # 3) Identify contiguous fill blocks
    fill_mask = df['combined_fill_events'] > 0
    block_id  = (fill_mask != fill_mask.shift()).cumsum()
    df['block_id'] = block_id.where(fill_mask)
    
    # 4) Compute mean fill volume per block
    fill_blocks       = df[fill_mask].copy()
    fill_block_means  = fill_blocks.groupby('block_id')['combined_fill_events'].mean()
    
    # 5) Assign corrected volumes at block start
    block_starts = (
        fill_blocks
        .groupby('block_id')
        .head(1)
        .assign(Fluid_Added_Corrected = fill_block_means.values)
    )
    block_starts['Cumulative_Fluid_Added_Corrected'] = block_starts['Fluid_Added_Corrected'].cumsum()
    
    # 6) Merge corrected volumes back and forward‐fill
    df['Fluid_Added_Corrected'] = block_starts['Fluid_Added_Corrected']
    df['Cumulative_Fluid_Added_Corrected'] = block_starts['Cumulative_Fluid_Added_Corrected']
    df[['Fluid_Added_Corrected','Cumulative_Fluid_Added_Corrected']] = \
        df[['Fluid_Added_Corrected','Cumulative_Fluid_Added_Corrected']].ffill().fillna(0)
        
    return df
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
def system_fluid_consumption(
    df: pd.DataFrame,
    level_col: str = 'HPU_SPLY_LEV_L',
    min_drop_threshold: float = 0.0,      # Litres
    smoothing_window: int = 0           # Rolling average window size (e.g. 5 minutes)
) -> pd.DataFrame:
    """
    Adds three columns to df:
      - Supply_Δ: per-step change (after smoothing)
      - Supply_Consumption_Excl_Fills: significant drops only (positive litres)
      - Cumulative_Supply_Consumption_Excl_Fills: cumulative sum of valid drops

    Parameters:
      df                : DataFrame with datetime index and a tank level column
      level_col         : name of the supply tank level column (litres)
      min_drop_threshold: minimum drop to count as consumption (litres)
      smoothing_window  : window size for moving average smoothing (set to 1 to disable)

    Returns:
      Updated DataFrame with 3 new columns.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors='coerce')
    df.sort_index(inplace=True)

    # Apply moving average smoothing (if enabled)
    if smoothing_window > 1:
        df['Smoothed_Level'] = df[level_col].rolling(window=smoothing_window, min_periods=1).mean()
    else:
        df['Smoothed_Level'] = df[level_col]

    # Compute change in smoothed tank level
    df['Supply_Δ'] = df['Smoothed_Level'].diff()

    # Only significant negative changes (drops), ignoring noise
    drops = df['Supply_Δ'].clip(upper=0).abs()
    df['Supply_Consumption_Excl_Fills'] = drops.where(drops >= min_drop_threshold, 0)

    # Running total of consumption
    df['Cumulative_Supply_Consumption_Excl_Fills'] = df['Supply_Consumption_Excl_Fills'].cumsum()

    return df
################################################################################################################################################################

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
def compute_baseline_drift(
    df: pd.DataFrame,
    lp_col: str = 'SCM1_LP_CONS',
    hp_col: str = 'SCM1_HP_CONS',
    tank_col: str = 'HPU_SPLY_LEV_L',
    fcv_steps_col: str = 'FCV_FullSteps',
    valve_transition_cols: list = [
        '2_LP_Valve_OpenToClosed', '2_LP_Valve_ClosedToOpen',
        '5_LP_Valve_OpenToClosed', '5_LP_Valve_ClosedToOpen',
        'HP_Valve_OpenToClosed',  'HP_Valve_ClosedToOpen'
    ],
    min_lp: float = 180,
    min_hp: float = 200,
    clean_duration: pd.Timedelta = pd.Timedelta('12h')
) -> (pd.DataFrame, float):
    """
    Identifies “clean” 12‑hour periods with no valve/FCV activity and
    adequate pressures, computes the tank‐level drop over each period,
    and fits a linear trend (L/day).

    Returns:
      baseline_df: DataFrame with columns ['start','end','drop_L']
      slope_lday : float slope in L/day of baseline drift
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors='coerce')
    df.sort_index(inplace=True)
    
    # 1) cast to numeric
    df['LP']   = pd.to_numeric(df[lp_col], errors='coerce')
    df['HP']   = pd.to_numeric(df[hp_col], errors='coerce')
    df['Tank'] = pd.to_numeric(df[tank_col], errors='coerce')
    df['FCV_steps'] = (
        pd.to_numeric(df[fcv_steps_col], errors='coerce')
          .fillna(0)
          .astype(int)
    )
    
    # 2) clean‐period mask
    mask = (
        (df['LP'] > min_lp) &
        (df['HP'] > min_hp) &
        (df['FCV_steps'] == 0) &
        (df[valve_transition_cols].sum(axis=1) == 0)
    )
    
    # 3) run‐length encode
    df['run_id'] = (mask != mask.shift()).cumsum()
    
    # 4) scan runs
    periods = []
    for run, grp in df[mask].groupby('run_id'):
        start = grp.index[0]
        end   = start + clean_duration
        if end <= grp.index[-1]:
            lvl0 = df.at[start, 'Tank']
            lvl1 = df['Tank'].asof(end)
            drop = lvl0 - lvl1
            periods.append({'start': start, 'end': end, 'drop_L': drop})
    baseline_df = pd.DataFrame(periods)
    if baseline_df.empty:
        return baseline_df, np.nan
    
    # 5) fit linear trend in L/day
    x = mdates.date2num(baseline_df['start'])
    y = baseline_df['drop_L'].astype(float).values
    slope, intercept = np.polyfit(x, y, 1)
    # slope is in L per matplotlib day → convert to L/day by multiplying by 1
    slope_lday = slope
    
    return baseline_df, slope_lday
################################################################################################################################################################

def add_baseline_columns(
    df: pd.DataFrame,
    **drift_kwargs
) -> pd.DataFrame:
    """
    1) Runs compute_baseline_drift(df, **drift_kwargs)
    2) Adds `baseline_drop_L` (non-NaN only at each 12h-clean window start)
    3) Adds `baseline_drift_L_per_day` (constant slope) to every row
    
    Returns the augmented DataFrame.
    """
    # --- 1) compute baseline drops + slope ---
    baseline_df, slope_lday = compute_baseline_drift(df, **drift_kwargs)
    
    # --- 2) map start→drop_L into a dict ---
    # If there are no baseline periods, baseline_df will be empty
    drop_map = {}
    if not baseline_df.empty:
        drop_map = dict(zip(baseline_df['start'], baseline_df['drop_L']))
    
    # --- 3) build the output DF ---
    df_out = df.copy()
    
    # Map each timestamp: if it's one of the 'start' times, get the drop; else NaN
    df_out['baseline_drop_L'] = df_out.index.map(drop_map)
    
    # Fill constant drift
    df_out['baseline_drift_L_per_day'] = slope_lday
    
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
    df_all_otter,
    pump_run_csv,
    pumps=("LP_Pump_A", "HP_Pump_A"),
    count_col_template="Cum_{pump}_Run_Count",
    dur_col_template="Cum_{pump}_Run_Dur",
    time_col="Start Time",
    duration_col="Duration (min)",
    freq="T"
):
    """
    Reads pump‐run events from CSV and adds four cumulative columns to df_all_otter:
      • Cum_<pump>_Run_Count   (cumulative number of runs)
      • Cum_<pump>_Run_Dur     (cumulative run duration in minutes)

    Parameters
    ----------
    df_all_otter : DataFrame
        Your main DataFrame; its index will be coerced to datetime & sorted.
    pump_run_csv : str
        Path to the pump_run_events.csv file (must have 'Pump', 'Start Time', 'Duration (min)').
    pumps : tuple of two str
        The values in the 'Pump' column you wish to track (e.g. ("LP_Pump_A","HP_Pump_A")).
    count_col_template : str
        Template for the count column name; use "{pump}" to interpolate.
    dur_col_template : str
        Template for the duration column name; use "{pump}" to interpolate.
    time_col : str
        Name of the CSV column for the run start timestamp.
    duration_col : str
        Name of the CSV column for the run duration (in minutes).
    freq : str
        Resampling frequency for the time series (default "T" = minute).
    """
    # 1) Prepare main DF
    df = df_all_otter.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df.sort_index(inplace=True)

    # 2) Load runs CSV
    runs = pd.read_csv(pump_run_csv, parse_dates=[time_col])

    # 3) For each pump, build and reindex cum‐count & cum‐dur series
    for pump in pumps:
        rp = runs[runs["Pump"] == pump]

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
    df_all_otter,
    slope_parquet_path
):
    """
    Read slope features from a Parquet file, select drop-slope columns,
    and append them to df_all_otter, after dropping duplicates from slope_df.

    Steps:
      1. Load slope_df from Parquet.
      2. Ensure datetime index and sort.
      3. Drop duplicate timestamps (keep first).
      4. Select only the drop-slope related columns.
      5. Reset index with consistent name.
      6. Remove any overlapping columns from df_all_otter.
      7. Merge slope_df onto df_all_otter and return.

    Parameters
    ----------
    df_all_otter : pd.DataFrame
        Main time‑indexed DataFrame.
    slope_parquet_path : str
        Path to the Parquet file containing slope features.

    Returns
    -------
    pd.DataFrame
        Augmented df_all_otter with slope feature columns appended.
    """
    # 1) Load slope features
    slope_df = pd.read_parquet(slope_parquet_path)

    # 2) Ensure datetime index & sorted
    slope_df.index = pd.to_datetime(slope_df.index, errors="coerce")
    slope_df.sort_index(inplace=True)

    # 3) Drop duplicate timestamps
    before = len(slope_df)
    slope_df = slope_df[~slope_df.index.duplicated(keep='first')]
    dropped = before - len(slope_df)
    if dropped:
        print(f"Dropped {dropped} duplicate rows from slope_df.")

    # 4) Select only drop-slope related columns
    drop_slope_cols = [
        'Slope_2H_Lph', 'Slope_12H_Lph', 'Slope_24H_Lph','Slope_1H_Lph', 'Slope_7D_Lph',
        'HPU_LPA_OUT_DropSlope_1h_Lph','HPU_LPA_OUT_DropSlope_2h_Lph', 'HPU_LPA_OUT_DropSlope_8h_Lph',
         'HPU_LPB_OUT_DropSlope_1h_Lph','HPU_LPB_OUT_DropSlope_2h_Lph', 'HPU_LPB_OUT_DropSlope_8h_Lph',
        'HPU_HPA_OUT_DropSlope_1h_Lph','HPU_HPA_OUT_DropSlope_2h_Lph', 'HPU_HPA_OUT_DropSlope_8h_Lph',
         'HPU_HPB_OUT_DropSlope_1h_Lph','HPU_HPB_OUT_DropSlope_2h_Lph', 'HPU_HPB_OUT_DropSlope_8h_Lph'
    ]
    slope_df = slope_df[drop_slope_cols]

    # 5) Reset index with consistent name
    if 'Timestamp' in slope_df.columns:
        slope_df.rename(columns={'Timestamp': 'timestamp'}, inplace=True)
    slope_df.index.name = 'timestamp'
    df_all_otter.index.name = 'timestamp'
    df_all_otter = df_all_otter.reset_index()
    slope_df = slope_df.reset_index()

    # 6) Drop overlapping columns if needed
    overlapping = df_all_otter.columns.intersection(slope_df.columns)
    overlapping = overlapping.drop('timestamp', errors='ignore')
    if not overlapping.empty:
        print(f"Removing overlapping columns from df_all_otter: {list(overlapping)}")
        df_all_otter = df_all_otter.drop(columns=overlapping)

    # 7) Merge on timestamp
    df_aug = pd.merge(df_all_otter, slope_df, how='left', on='timestamp')

    # 8) Restore datetime index
    df_aug.set_index('timestamp', inplace=True)
    df_aug.index.name = None

    # 9) Report missing timestamps
    missing = df_aug[drop_slope_cols].isnull().all(axis=1).sum()
    if missing:
        print(f"Warning: {missing} timestamps in df_all_otter had no slope data (NaN inserted).")

    return df_aug
################################################################################################################################################################
def clean_dataframe(df):
    # 1. Columns to drop
    cols_to_drop = [
        'HPU_SPLY_LEV_B', 'HPU-RET_LEV_B', 'P1_PMV_OpenToClosed', 'P1_PMV_ClosedToOpen', 'P1_PWV_OpenToClosed',
        'P1_PWV_ClosedToOpen', 'P1_AMV_OpenToClosed', 'P1_AMV_ClosedToOpen', 'P1_SCSSV_OpenToClosed',
        'P1_SCSSV_ClosedToOpen', 'P1_PDV_OpenToClosed', 'P1_PDV_ClosedToOpen', 'P1_TDV_OpenToClosed',
        'P1_TDV_ClosedToOpen', 'P2_PMV_OpenToClosed', 'P2_PMV_ClosedToOpen', 'P2_PWV_OpenToClosed',
        'P2_PWV_ClosedToOpen', 'P2_AMV_OpenToClosed', 'P2_AMV_ClosedToOpen', 'P2_SCSSV_OpenToClosed',
        'P2_SCSSV_ClosedToOpen', 'P2_PDV_OpenToClosed', 'P2_PDV_ClosedToOpen', 'P2_TDV_OpenToClosed',
        'P2_TDV_ClosedToOpen', 'P3_PMV_OpenToClosed', 'P3_PMV_ClosedToOpen', 'P3_PWV_OpenToClosed',
        'P3_PWV_ClosedToOpen', 'P3_AMV_OpenToClosed', 'P3_AMV_ClosedToOpen', 'P3_SCSSV_OpenToClosed',
        'P3_SCSSV_ClosedToOpen', 'P3_PDV_OpenToClosed', 'P3_PDV_ClosedToOpen', 'P3_TDV_OpenToClosed',
        'P3_TDV_ClosedToOpen', 'I1_PMV_OpenToClosed', 'I1_PMV_ClosedToOpen', 'I1_PWV_OpenToClosed',
        'I1_PWV_ClosedToOpen', 'I1_AMV_OpenToClosed', 'I1_AMV_ClosedToOpen', 'I2_PMV_OpenToClosed',
        'I2_PMV_ClosedToOpen', 'I2_PWV_OpenToClosed', 'I2_PWV_ClosedToOpen', 'I2_AMV_OpenToClosed',
        'I2_AMV_ClosedToOpen', 'MPMV_Inlet_OpenToClosed', 'MPMV_Inlet_ClosedToOpen', 'Man_CI_OpenToClosed',
        'Man_CI_ClosedToOpen', 'SCM1_LP1_COV_OpenToClosed', 'SCM1_LP1_COV_ClosedToOpen', 'SCM1_HP_COV_OpenToClosed',
        'SCM1_HP_COV_ClosedToOpen'
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # 2. Move HPU_SPLY_LEV_L and HPU_RET_LEV_L to the front
    lead_cols = ['HPU_SPLY_LEV_L', 'HPU_RET_LEV_L']
    existing_lead_cols = [col for col in lead_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in existing_lead_cols]
    df = df[existing_lead_cols + other_cols]

    # 3. Rename slope columns
    slope_rename_map = {
        'Slope_1H_Lph': 'Supply_Slope_1H_Lph',
        'Slope_2H_Lph': 'Supply_Slope_2H_Lph',
        'Slope_12H_Lph': 'Supply_Slope_12H_Lph',
        'Slope_24H_Lph': 'Supply_Slope_24H_Lph',
        'Slope_7D_Lph': 'Supply_Slope_7D_Lph'
    }

    # 4. Rename MA/EWM columns
    external_rename_map = {
        'MA_2h': 'External_Loss_MA_2h',
        'MA_12h': 'External_Loss_MA_12h',
        'MA_24h': 'External_Loss_MA_24h',
        'MA_168h': 'External_Loss_MA_168h',
        'EWM_1d': 'External_Loss_EWM_1d',
        'EWM_7d': 'External_Loss_EWM_7d',
        'EWM_30d': 'External_Loss_EWM_30d'
    }

    # Combine and apply renaming
    rename_map = {**slope_rename_map, **external_rename_map}
    df = df.rename(columns=rename_map)

    return df
################################################################################################################################################################


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
    drop_slope_data_path: str
) -> pd.DataFrame:
    """
    Runs the full Otter hydraulic data pipeline and returns a cleaned, feature-rich DataFrame.
    """
    df = load_and_clean_otter_data(parquet_path, columns_to_remove)

    df = process_valve_data(df, valve_columns)

    df = valve_fluid_usage_calc(
        df,
        fluid_per_5lp=valve5_volume,
        fluid_per_2lp=valve2_volume,
        fluid_per_hp=valve_hp_volume
    )

    df = process_fcv(df)
    df = process_pcv(df, pcv_columns)
    df = convert_tank_levels(df)

    df = process_umbilical_charges(
        df,
        channel_thresholds=thresholds
    )

    df = system_fluid_consumption(df)
    df = add_external_losses(df)
    df = add_external_loss_moving_averages(df)

    df = add_daily_ewm_to_minutely_df(
        df,
        loss_col='External_Losses',
        spans=[1, 7, 30]
    )

    df = add_baseline_columns(
        df,
        lp_col='SCM1_LP_CONS',
        hp_col='SCM1_HP_CONS',
        tank_col='HPU_SPLY_LEV_L',
        fcv_steps_col='FCV_FullSteps',
        valve_transition_cols=valve_transition_cols,
        min_lp=180,
        min_hp=200,
        clean_duration=pd.Timedelta('12h')
    )

    df = add_pressure_deltas(df)

    df = add_pump_run_cumulatives(
        df,
        pump_events_path,
        pumps=("LP_Pump_A", "HP_Pump_A")
    )

    df = add_slope_features(df, drop_slope_data_path)
  

    return df
################################################################################################################


