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
