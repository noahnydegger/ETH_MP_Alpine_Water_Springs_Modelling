import pandas as pd  # data processing
import numpy as np  # data processing
from datetime import datetime, timedelta
import pathlib
import pickle


def get_catchment_parameters(spring_name):
    # Catchment properties
    if spring_name == 'Ulrika':
        fixed_parameters = {
            'latitude': 47.3,  # mean Latitude [deg]
            'elevation_difference': 750 - 647  # mean elevation [masl]
        }
    elif spring_name == 'Paliu_Fravi':
        fixed_parameters = {
            'latitude': 46.8,  # mean Latitude [deg]
            'elevation_difference': 900 - 562  # mean elevation [masl]
        }

    fixed_parameters['melting_temperature'] = 0.0,  # Temperature threshold for melting [deg Celsius]
    fixed_parameters['temperature_lapse_rate'] = -0.5 / 100  # Lapse rate for temperature [deg Celsius/1m elevation]
    fixed_parameters['melting_rate'] = 0.8  # Daily degree-day snow melt parameter [mm/day/degC]

    return fixed_parameters


def get_model_parameters(spring_name):
    # Model parameters_for_df: best parameters_for_df so far
    if spring_name == 'Ulrika':
        variable_parameters = {
            'area': 127091.2775,#194854.08,# 389459,  # estimated area [m ^ 2]
            'storage_capacity': 5.0208,  # Soil zone storage capacity [mm]
            'residence_time': 11.9657,#26.0,  # Ground water linear reservoir constant, mean residence time [days]
            'runoff_fraction': 0.01027,  # Fraction of basin area that is generating surface runoff [0-1]
            'melting_rate': 5.8371#0.8  # Daily degree-day snow melt parameter [mm/day/degC]
        }
    elif spring_name == 'Paliu_Fravi':
        variable_parameters = {
            'area': 181820.3138,  # estimated area [m ^ 2]
            'storage_capacity': 13.3589,  # Soil zone storage capacity [mm]
            'residence_time': 26.9823,  # Ground water linear reservoir constant, mean residence time [days]
            'runoff_fraction': 0.0991,  # Fraction of basin area that is generating surface runoff [0-1]
            'melting_rate': 1.8522#0.8  # Daily degree-day snow melt parameter [mm/day/degC]
        }

    return variable_parameters


def create_rain_and_snow_input(wb_df, fixed_parameters):
    # compute rain fall and snow fall with hourly resolution depending on snow melt temperature
    melt_rate = fixed_parameters['melting_rate']
    melt_temp = fixed_parameters['melting_temperature']
    wb_df['rain_fall(mm)'] = np.where(wb_df['temperature(C)'] >= melt_temp, wb_df['precipitation(mm)'], 0)
    wb_df['snow_fall(mm)'] = np.where(wb_df['temperature(C)'] < melt_temp, wb_df['precipitation(mm)'], 0)

    # set columns initially to zero
    num_rows = len(wb_df)  # Get the number of rows from the existing DataFrame
    wb_df['snow_melt(mm)'] = [0.0] * num_rows  # snow melt in mm water equivalent
    wb_df['snow_cover(mm)'] = [0.0] * num_rows  # snow cover in mm water equivalent

    return wb_df


def create_potential_evapotranspiration_input(wb_df, fixed_parameters):
    lat = fixed_parameters['latitude']

    # calculate the pet parameters_for_df
    wb_df['delta'] = 0.4093 * np.sin((2 * np.pi / 365) * wb_df['doy'] - 1.405)
    wb_df['omega_s'] = np.arccos(-np.tan(2 * np.pi * lat / 360) * np.tan(wb_df['delta']))
    wb_df['daytime_hours'] = 24 * wb_df['omega_s'] / np.pi
    wb_df['sat_vap_pres'] = 0.6108 * np.exp(17.27 * wb_df['temperature(C)'] / (wb_df['temperature(C)'] + 237.3))
    wb_df['pET(mm)'] = np.where(wb_df['temperature(C)'] < 0, 0,
                                2.1 * (wb_df['daytime_hours']**2) * wb_df['sat_vap_pres']) / (wb_df['temperature(C)'] + 273.3)


    # remove the pet parameter columns
    wb_df.drop(columns=['delta', 'omega_s', 'daytime_hours', 'sat_vap_pres'], inplace=True)

    return wb_df


def create_model_parameter_columns(wb_df):
    num_rows = len(wb_df)  # Get the number of rows from the existing DataFrame
    # Add columns for model variables with initial values of 0
    wb_df['aET(mm)'] = [0.0] * num_rows  # actual evapotranspiration
    wb_df['storage_soil(mm)'] = [0.0] * num_rows  # water saturation in the soil reservoir
    wb_df['storage_gw(mm)'] = [0.0] * num_rows  # water storage in the groundwater reservoir
    wb_df['runoff(mm)'] = [0.0] * num_rows  # surface runoff
    wb_df['percolation_gw(mm)'] = [0.0] * num_rows  # percolation from saturated soil to groundwater reservoir
    wb_df['discharge_sim(mm)'] = [0.0] * num_rows  # spring discharge simulated

    return wb_df


def resample_model_input_to_daily(wb_df, fixed_parameters):
    wb_df_D = wb_df.resample('D').agg({'doy': 'last',
                                       'valid_spring': 'all',
                                       'valid_meteo': 'all',
                                       'valid': 'all',
                                       'optimization_period': 'first',
                                       'calibration_period': 'first',
                                       'validation_period': 'first',
                                       'final_validation_period': 'first',
                                       'discharge_meas(L/min)': 'mean',
                                       'precipitation(mm)': 'sum',
                                       'rain_fall(mm)': 'sum',
                                       'snow_fall(mm)': 'sum',
                                       'snow_melt(mm)': 'sum',
                                       'snow_cover(mm)': 'last'
                                       })
    # add a column with mean temperature
    # Resample temperature to daily minimum and maximum
    temp_resampled = wb_df['temperature(C)'].resample('D').agg(['min', 'max'])
    # Calculate the mean temperature as the daily average of min and max
    mean_temp = temp_resampled.mean(axis=1)
    # Add the temperature column at a specific position
    wb_df_D.insert(9, 'temperature(C)', mean_temp)

    wb_df_D = create_potential_evapotranspiration_input(wb_df_D, fixed_parameters)

    wb_df_D = create_model_parameter_columns(wb_df_D)

    return wb_df_D


def initialize_wb_df(spring_name, fixed_parameters, path_to_data_folder):
    var = 'discharge'
    pkl_file_path = path_to_data_folder / 'water_balance_model' / spring_name / f'wb_{spring_name}_{var}.pkl'
    wb_df = pd.read_pickle(pkl_file_path)
    wb_df.rename(columns={'discharge(L/min)': 'discharge_meas(L/min)', 'valid': 'valid_spring'}, inplace=True)
    wb_df['valid_meteo'] = True

    for var in ['temperature', 'precipitation']:
        pkl_file_path = path_to_data_folder / 'water_balance_model' / spring_name / f'wb_{spring_name}_{var}.pkl'
        var_data = pd.read_pickle(pkl_file_path)
        var_data.rename(columns={'valid': f'valid_{var}'}, inplace=True)
        # merge new variable data
        wb_df = wb_df.merge(var_data, how='outer', left_index=True,
                            right_index=True)
        # set new dates to false
        wb_df['valid_spring'].fillna(False, inplace=True)

        # set valid_meteo to false where valid_var is false
        wb_df['valid_meteo'] = np.where(wb_df[f'valid_{var}'], wb_df['valid_meteo'], False)
        # remove the row containing valid for the current variable
        wb_df.drop(columns=f'valid_{var}', inplace=True)

    # make a columm for the calculation of the gof
    wb_df['valid'] = wb_df['valid_spring'] & wb_df['valid_meteo']
    wb_df['doy'] = wb_df.index.dayofyear  # Julien day = day of the year

    # correct the temperature with the elevation difference between the catchment and the main meteo station
    elev_diff = fixed_parameters['elevation_difference']
    lapse_rate = fixed_parameters['temperature_lapse_rate']
    wb_df['temperature(C)'] = wb_df['temperature(C)'] + elev_diff * lapse_rate

    # set calibration, validation periods
    if spring_name == 'Ulrika':
        # calibration # data gap between 2021-08-16 to 2021-10-13 and 2022-05-02 to 2022-08-02
        start_date_cal = wb_df.index[wb_df['valid']].min() - pd.DateOffset(months=1)
        end_date_cal = pd.to_datetime('2022-09-15').tz_localize('UTC')
        # validation
        start_date_val = end_date_cal + pd.Timedelta(days=1)
        end_date_val = pd.to_datetime('2023-08-31').tz_localize('UTC')
        # final validation
        start_date_final = end_date_val + pd.Timedelta(days=1)
    elif spring_name == 'Paliu_Fravi':
        # calibration
        start_date_cal = pd.to_datetime('2021-10-25').tz_localize('UTC')
        end_date_cal = pd.to_datetime('2022-8-31').tz_localize('UTC')  # valid until 2022-08-29
        # validation
        start_date_val = end_date_cal + pd.Timedelta(days=1)  # valid from 2022-11-11
        end_date_val = pd.to_datetime('2023-08-31').tz_localize('UTC')
        # final validation
        start_date_final = end_date_val + pd.Timedelta(days=1)

    wb_df['calibration_period'] = (wb_df.index >= start_date_cal) & (wb_df.index <= end_date_cal)
    wb_df['validation_period'] = (wb_df.index >= start_date_val) & (wb_df.index <= end_date_val)
    wb_df['final_validation_period'] = (wb_df.index >= start_date_final)

    # add a column specifying the date range for the optimization
    # use data until end of August for the optimization, keep September for testing
    # optimization period
    wb_df['optimization_period'] = (wb_df.index >= start_date_cal) & (wb_df.index <= end_date_val)

    return wb_df


def create_model_input_df(spring_name, fixed_parameters, path_to_data_folder, resolution):
    # create a dataframe to store the water balance model timeseries with hourly resolution
    wb_df = initialize_wb_df(spring_name, fixed_parameters, path_to_data_folder)

    # adds column with rain fall, snow fall, snow cover, snow melt
    wb_df = create_rain_and_snow_input(wb_df, fixed_parameters)

    # adds empty columns for the model calculations
    wb_df = create_model_parameter_columns(wb_df)

    if resolution == 'D':
        wb_df = resample_model_input_to_daily(wb_df, fixed_parameters)

    return wb_df


def initialize_model(spring_name, path_to_data_folder, resolution, optimize):
    if not spring_name in ['Ulrika', 'Paliu_Fravi']:
        raise ValueError(f"expected 'Ulrika' or 'Paliu_Fravi' as spring_name but got {spring_name}.")

    if not resolution in ['H', 'D']:
        raise ValueError(f"expected 'H' or 'D' as resolution but got {resolution}.")

    fixed_parameters = get_catchment_parameters(spring_name)
    variable_parameters = get_model_parameters(spring_name)

    wb_df_name = f'wb_{spring_name}_input_{resolution}.pkl'
    wb_df_path = path_to_data_folder / 'water_balance_model' / spring_name / wb_df_name

    if wb_df_path.exists():
        # Load the DataFrame from the pickle file
        with open(wb_df_path, 'rb') as file:
            wb_df = pickle.load(file)
        print(f'{wb_df_name} loaded')
    else:
        wb_df = create_model_input_df(spring_name, fixed_parameters, path_to_data_folder, resolution)
        wb_df.to_pickle(wb_df_path)
        wb_df.to_csv(wb_df_path.with_suffix('.csv'))
        print(f'{wb_df_name} created')

    # select subsection for faster optimization
    if optimize:
        # Drop rows where column 'B' is False in-place
        wb_df.drop(wb_df[~wb_df['optimization_period']].index, inplace=True)

    return fixed_parameters, variable_parameters, wb_df
