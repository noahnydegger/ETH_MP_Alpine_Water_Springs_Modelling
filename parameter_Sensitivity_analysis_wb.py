from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import Data
from Water_Balance_Model import wb_Initialization
from Water_Balance_Model import wb_Computation
from Water_Balance_Model import wb_Visualization
from datetime import datetime
import time

resolution = 'D'

for spring_name in ['Ulrika', 'Paliu_Fravi']:
    start_time = time.time()
    # initialize the model
    fixed_parameters, variable_parameters, wb_df = wb_Initialization.initialize_model(spring_name, Data.datafolder,
                                                                                      resolution, optimize=True)
    # select subset for calibration
    wb_df = wb_df[wb_df['calibration_period']].copy()

    # create parameter range
    # get the optimal parameters
    params_opt_list = list(variable_parameters.values())
    area_opt = params_opt_list[0]
    smax_opt = params_opt_list[1]
    r_opt = params_opt_list[3]
    # set the range for the sensitivity analysis
    area_range = np.arange(area_opt * 0.8, area_opt * 1.22, area_opt * 0.02) #1.21
    if spring_name == 'Ulrika':
        smax_range = np.arange(1, 41, 1)
        rg_range = np.arange(5, 15.5, 0.5)
    elif spring_name == 'Paliu_Fravi':
        smax_range = np.arange(1, 41, 1)
        rg_range = np.arange(10, 31, 1)
    fr_range = np.arange(0.0, 0.16, 0.01)
    #nr_combinations = len(area_range) * len(smax_range) * len(fr_range)
    nr_combinations = len(smax_range) * len(rg_range)
    i = 0
    curr_params = params_opt_list
    parameters_for_df = []
    gof_for_df = []
    #for area in area_range:
    #    curr_params[0] = area
    for smax in smax_range:
        curr_params[1] = smax
        for rg in rg_range:  # fr
            curr_params[2] = rg  # fr
            # run the model
            wb_df, gof_values = wb_Computation.compute_water_balance_iterative(wb_df,
                                                                               fixed_parameters,
                                                                               curr_params,
                                                                               resolution)

            parameters_for_df.append(curr_params.copy())
            gof_for_df.append(list(gof_values.values()))

            # show current status
            i += 1
            curr_percentage = (i / nr_combinations * 100)
            if (curr_percentage % 5) < 0.012:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"{np.round(curr_percentage, 2)} % after {np.round(elapsed_time, 2)} "
                      f"seconds for spring: {spring_name}")

    # save the results to pickle and csv
    sensitivity_results_df = pd.DataFrame()
    param_names = list(variable_parameters.keys())
    sensitivity_results_df[param_names] = parameters_for_df
    sensitivity_results_df[list(gof_values.keys())] = gof_for_df
    # Get current date, hour, and minute
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    file_path = (Data.datafolder / 'water_balance_model' / f'Results_{spring_name}' / 'Sensitivity_Results' /
                 curr_time / f'{curr_time}_sensitivity_results_{spring_name}.pkl')
    file_path.parent.mkdir(parents=True, exist_ok=True)
    sensitivity_results_df.to_pickle(file_path)
    sensitivity_results_df.to_csv(file_path.with_suffix('.csv'), index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {np.round(elapsed_time, 2)} seconds for spring: {spring_name}")
