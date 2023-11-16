import pandas as pd  # data processing
import numpy as np  # data processing

from WatBalMod import WatBal_Initialization
from WatBalMod import WatBal_Computation
from WatBalMod import WatBal_Visualization


def run_WatBal_model(spring_name, meteo_name, path_to_data_folder):
    catchment_parameters, model_parameters, pet_Hamon_parameters = WatBal_Initialization.get_model_parameters()

    wb_df = WatBal_Initialization.create_model_input_df(spring_name, meteo_name, catchment_parameters, model_parameters,
                                                        path_to_data_folder)

    wb_df, gof_values = WatBal_Computation.compute_water_balance(wb_df, catchment_parameters, model_parameters)

    WatBal_Visualization.first_impression_plot(spring_name, meteo_name, wb_df, catchment_parameters)
    WatBal_Visualization.static_discharge_plot(spring_name, meteo_name, wb_df, catchment_parameters)

    return wb_df, gof_values
