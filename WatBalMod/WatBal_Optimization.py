import pandas as pd  # data processing
import numpy as np  # data processing
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.model_selection import GridSearchCV

import WatBal_Initialization
import WatBal_Computation
import WatBal_Visualization


def optimize_by_time_series_split(n_splits=2):
    catchment_parameters, model_parameters, pet_Hamon_parameters = WatBal_Initialization.get_model_parameters()

    wb_df = WatBal_Initialization.create_model_input_df(spring_name, meteo_name, catchment_parameters, model_parameters,
                                                        path_to_data_folder)

    tscv = TimeSeriesSplit(n_splits=n_splits)  # Adjust the number of splits as needed
    for i, train_index, test_index in enumerate(tscv.split(wb_df)):
        train_data, test_data = wb_df.iloc[train_index], wb_df.iloc[test_index]

        wb_df, gof_values = WatBal_Computation.compute_water_balance(wb_df, catchment_parameters, model_parameters)

        # Initialize and train your model
