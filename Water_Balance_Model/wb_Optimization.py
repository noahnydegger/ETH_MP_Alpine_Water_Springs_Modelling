import pandas as pd  # data processing
import numpy as np  # data processing
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy import optimize

import wb_Initialization
import wb_Computation
import wb_Visualization


def objective_function(variable_parameters, wb_df, tscv):
    total_fit_value = 0.0

    for train_index, test_index in tscv.split(wb_df):
        train_data, test_data = wb_df.iloc[train_index], wb_df.iloc[test_index]
        train_data, gof_values = WatBal_Computation.compute_water_balance(train_data, variable_parameters)

        total_fit_value += gof_values['RMSE']

    return total_fit_value






def optimize_by_time_series_split(n_splits=2):
    catchment_parameters, model_parameters, pet_Hamon_parameters = WatBal_Initialization.get_model_parameters()

    wb_df = WatBal_Initialization.create_model_input_df(spring_name, meteo_name, catchment_parameters, model_parameters,
                                                        path_to_data_folder)

    tscv = TimeSeriesSplit(n_splits=n_splits)  # Adjust the number of splits as needed

    parameter_bounds = []
    result = optimize.differential_evolution(objective_function, parameter_bounds, args=(wb_df, tscv))
    for i, train_index, test_index in enumerate(tscv.split(wb_df)):
        train_data, test_data = wb_df.iloc[train_index], wb_df.iloc[test_index]

        wb_df, gof_values = WatBal_Computation.compute_water_balance(wb_df, catchment_parameters, model_parameters)

        # Initialize and train your model
