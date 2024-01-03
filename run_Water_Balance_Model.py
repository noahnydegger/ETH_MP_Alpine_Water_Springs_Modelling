from pathlib import Path
from scipy import optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import Data
from Water_Balance_Model import wb_Initialization
from Water_Balance_Model import wb_Computation
from Water_Balance_Model import wb_Visualization
from datetime import datetime
import time

# specify the spring
spring_name = 'Ulrika'
resolution = 'D'
optimize_model = False
optimize_method = 'Basin'  # 'Basin', 'Evolution'
optimize_metric = 'NSE'  # 'NSE', 'RMSE'

# make beautiful plots
pp_plot = True
selected_time = {
    'Ulrika': '20231208_2300',
    'Paliu_Fravi': '20231209_0124'
}

# initialize the model
fixed_parameters, variable_parameters, wb_df = wb_Initialization.initialize_model(spring_name, Data.datafolder, resolution, optimize_model)
start_time = time.time()
if optimize_model:
    result, intermediate_results_df = wb_Computation.optimize_wb_model(spring_name, variable_parameters,
                                                                       fixed_parameters, wb_df, resolution,
                                                                       optimize_method, optimize_metric)

    # Export DataFrame to pickle and csv
    # Get current date, hour, and minute
    curr_time = datetime.now().strftime("%Y%m%d_%H%M")
    file_path = (Data.datafolder / 'water_balance_model' / f'Results_{spring_name}' /
                 f'{optimize_method}_{optimize_metric}' / f'{curr_time}_intermediate_results.pkl')
    file_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate_results_df.to_pickle(file_path)
    intermediate_results_df.to_csv(file_path.with_suffix('.csv'), index=False)

    # Save result as pickle
    result_pickle_filename = file_path.parent / f'{curr_time}_optimization_result.pkl'
    with open(result_pickle_filename, 'wb') as result_pickle_file:
        pickle.dump(result, result_pickle_file)

    # Retrieve optimal parameters_for_df (including the fixed parameter)
    optimal_parameters = result.x
    print(optimal_parameters)
    wb_df, gof_values = wb_Computation.compute_water_balance_iterative(wb_df, fixed_parameters, optimal_parameters, resolution)
    end_time = time.time()
    # create plots
    wb_Visualization.first_impression_plot(spring_name, wb_df)
else:
    if pp_plot:
        # get specific variable parameter values
        result_path = (Data.datafolder / 'water_balance_model' / f'Results_{spring_name}' / 'overall_Results' /
                       selected_time[spring_name] / f'{selected_time[spring_name]}_overall_results_{spring_name}.pkl')
        result_df = pd.read_pickle(result_path)
        variable_parameters = result_df.loc[result_df['NSE'].idxmax()][list(variable_parameters.keys())].to_dict()

        # select a subset of the wb_df
        #wb_df = wb_df[wb_df['calibration_period'] | wb_df['validation_period'] | wb_df['final_validation_period']]
        #wb_df = wb_df[wb_df.index > pd.Timestamp('2020-11-10', tz='UTC')]



    # single run
    wb_df, gof_values = wb_Computation.compute_water_balance_iterative(wb_df,
                                                                       fixed_parameters,
                                                                       list(variable_parameters.values()),
                                                                       resolution)
    end_time = time.time()
    print(gof_values)
    # create plots
    #wb_Visualization.first_impression_plot(spring_name, wb_df)

    # test plot
    gof_values_train = wb_Computation.compute_gof_values(wb_df[wb_df['calibration_period']])
    gof_values_test = wb_Computation.compute_gof_values(wb_df[wb_df['validation_period']])
    gof_values_val = wb_Computation.compute_gof_values(wb_df[wb_df['final_validation_period']])

    print('overall')
    print(gof_values)
    print('training')
    print(gof_values_train)
    print('testing')
    print(gof_values_test)
    print('validation')
    print(gof_values_val)
    # create plots

    #fig = wb_Visualization.overview_plot(spring_name, wb_df, gof_values_train, gof_values_test)
    fig = wb_Visualization.cal_val_period_plot(spring_name, wb_df, gof_values_train, gof_values_test, gof_values_val,
                                               gof_values)
    #fig = wb_Visualization.snow_plot(spring_name, wb_df, gof_values)
    fig_path = Data.datafolder / 'water_balance_model' / 'Plots' / spring_name / f'cal_val_plot_{spring_name}_all_long.pdf'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches='tight')
    plt.show()
    #plt.close(fig)



# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time} seconds")
