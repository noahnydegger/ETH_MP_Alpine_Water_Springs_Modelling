from pathlib import Path
from scipy import optimize
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
optimize_method = 'evolution'  # 'evolution', 'basin'
optimize_metric = 'NSE'  # 'NSE', 'RMSE'
nr_trials = 10
max_iter = 20

for spring_name in ['Ulrika', 'Paliu_Fravi']:
    overall_parameters = []
    overall_method = []
    # initialize the model
    fixed_parameters, variable_parameters, wb_df = wb_Initialization.initialize_model(spring_name, Data.datafolder,
                                                                                      resolution, optimize=True)

    for optimize_method in ['Evolution']:
        for optimize_metric in ['NSE']:
            for i in range(nr_trials):
                start_time = time.time()
                result, intermediate_results_df = wb_Computation.optimize_wb_model(spring_name, variable_parameters,
                                                                                   fixed_parameters, wb_df, resolution,
                                                                                   optimize_method, optimize_metric, max_iter)
                end_time = time.time()

                # save intermediate result to pickle and csv
                # Get current date, hour, and minute
                curr_time = datetime.now().strftime("%Y%m%d_%H%M")
                file_path = (Data.datafolder / 'water_balance_model' / f'Results_{spring_name}' /
                             f'{optimize_method}_{optimize_metric}' / curr_time /
                             f'{curr_time}_intermediate_results.pkl')
                file_path.parent.mkdir(parents=True, exist_ok=True)
                intermediate_results_df.to_pickle(file_path)
                intermediate_results_df.to_csv(file_path.with_suffix('.csv'), index=False)

                # Save result as pickle
                result_pickle_file_path = file_path.parent / f'{curr_time}_optimization_result.pkl'
                with open(result_pickle_file_path, 'wb') as result_pickle_file:
                    pickle.dump(result, result_pickle_file)

                # store result for the overall comparison
                overall_parameters.append(result.x)
                overall_method.append([optimize_method, optimize_metric, curr_time])

                # Retrieve optimal parameters_for_df (including the fixed parameter)
                optimal_parameters = result.x
                wb_df, gof_values = wb_Computation.compute_water_balance_iterative(wb_df, fixed_parameters,
                                                                                   optimal_parameters,
                                                                                   resolution)
                gof_values_cal = wb_Computation.compute_gof_values(wb_df[wb_df['calibration_period']])
                gof_values_val = wb_Computation.compute_gof_values(wb_df[wb_df['validation_period']])
                # create plots
                fig = wb_Visualization.overview_plot(spring_name, wb_df, gof_values_cal, gof_values_val)
                fig_path = file_path.parent / f'{curr_time}_{optimize_method}_{optimize_metric}.pdf'
                fig.savefig(fig_path, bbox_inches='tight')
                plt.close(fig)

                print(optimal_parameters)
                print(f"NSE: {np.round(gof_values_cal['NSE'], 2)}, RMSE: {np.round(gof_values_cal['RMSE'], 2)}")

                elapsed_time = end_time - start_time
                print(
                    f"Time taken: {np.round(elapsed_time, 2)} seconds for method: {optimize_method} with {optimize_metric} nr: {i + 1}")
                i += 1

    # save the overall result
    # create a DataFrame for the overall results
    param_names = list(variable_parameters.keys())
    overall_results_df = pd.DataFrame()

    overall_results_df[['method', 'metric', 'time']] = overall_method
    overall_results_df[param_names] = overall_parameters
    overall_gofs = []
    for xk in overall_parameters:
        wb_df, gof_values = wb_Computation.compute_water_balance_iterative(wb_df[wb_df['calibration_period']].copy(),
                                                                           fixed_parameters, xk, resolution)
        overall_gofs.append(list(gof_values.values()))
    overall_results_df[list(gof_values.keys())] = overall_gofs
    # save result to pickle and csv
    # Get current date, hour, and minute

    file_path = (Data.datafolder / 'water_balance_model' / f'Results_{spring_name}' / 'overall_Results' /
                 curr_time / f'{curr_time}_overall_results_{spring_name}.pkl')
    file_path.parent.mkdir(parents=True, exist_ok=True)
    overall_results_df.to_pickle(file_path)
    overall_results_df.to_csv(file_path.with_suffix('.csv'), index=False)
