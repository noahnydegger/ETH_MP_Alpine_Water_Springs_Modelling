import pandas as pd  # data processing
import numpy as np  # data processing
from datetime import datetime, timedelta
import os


def get_catchment_parameters(spring_name):
    # Catchment properties
    if spring_name == 'Ulrika':
        fixed_parameters = {
            'latitude': 46.5,  # mean Latitude [deg]
            'elevation': 1590  # mean elevation [masl]
        }
    elif spring_name == 'Paliu_Fravi':
        fixed_parameters = {
            'latitude': 46.5,  # mean Latitude [deg]
            'elevation': 1590  # mean elevation [masl]
        }
    return fixed_parameters


def get_model_parameters():
    # Model parameters
    variable_parameters = {
        'area': 389459,  # estimated area [m ^ 2]
        'storage_capacity': 15.0,  # Soil zone storage capacity [mm]
        'residence_time': 10.0,  # Ground water linear reservoir constant, mean residence time [days]
        'melting_rate': 0.8,  # Daily degree-day snow melt parameter [mm/day/degC]
        'runoff_fraction': 0.05,  # Fraction of basin area that is generating surface runoff [0-1]
        'melting_temperature': 0.0,  # Temperature threshold for melting [deg Celsius]
        'temperature_lapse_rate': 0.5 / 100  # Lapse rate for temperature [deg Celsius/1m elevation]
    }

    # Hamon PET parameters (constant, do not change):
    pet_Hamon_parameters = {
        'PETa': 0.6108,
        'PETb': 17.27,
        'PETc': 237.3
    }

    return variable_parameters, pet_Hamon_parameters


def initialize_wb_df(spring_name, path_to_data_folder, optimize):
    var = 'discharge'
    pkl_file_path = path_to_data_folder / 'water_balance_model' / spring_name / f'wb_{spring_name}_{var}.pkl'
    wb_df = pd.read_pickle(pkl_file_path)

    for var in ['temperature', 'precipitation']:
        pkl_file_path = path_to_data_folder / 'water_balance_model' / spring_name / f'wb_{spring_name}_{var}.pkl'
        var_data = pd.read_pickle(pkl_file_path)
        var_data.rename(columns={'valid': f'valid_{var}'}, inplace=True)
        if optimize:
            wb_df = wb_df.merge(var_data, how='left', left_index=True,
                        right_index=True)
            wb_df['valid'] = np.where(wb_df[f'valid_{var}'], wb_df['valid'], False)
        else:
            wb_df = wb_df.merge(var_data, how='outer', left_index=True,
                                right_index=True)
            wb_df['valid'].fillna(False, inplace=True)
            wb_df['valid'] = np.where(wb_df[f'valid_{var}'], wb_df['valid'], False)

        wb_df.drop(columns=f'valid_{var}', inplace=True)

    return wb_df


def create_spring_input(spring_name, path_to_data_folder, variable_parameters, resolution='D'):
    # load spring_name data
    spring_data = import_data_from_csv_file(find_file_by_name(f'{spring_name}_{resolution}',
                                                              path_to_data_folder, 'csv'))

    # Create a DataFrame to store model variables
    wb_df = pd.DataFrame(index=spring_data.index)
    wb_df['doy'] = wb_df.index.dayofyear  # Julien day = day of the year
    wb_df['valid'] = True  # set to False later, where spring or meteo data is missing

    wb_df['discharge_meas(L/min)'] = spring_data['discharge(L/min)']

    # exclude (set valid to False) data gaps and invalid measurements from the calibration period
    wb_df['valid'] = np.where(wb_df['discharge_meas(L/min)'].isna(), False, wb_df['valid'])
    wb_df['valid'] = np.where(wb_df['discharge_meas(L/min)'] < 5.0, False, wb_df['valid'])
    if spring_name == 'Ulrika':
        wb_df['valid'] = np.where(wb_df['discharge_meas(L/min)'] > 2000.0, False, wb_df['valid'])
    return wb_df


def create_temperature_input(wb_df, meteo_name, path_to_data_folder, resolution):

    # load temperature data
    temp_data = import_data_from_csv_file(find_file_by_name(f'{meteo_name}_temp_H',
                                                            path_to_data_folder, 'csv'))

    # add a column with min temperature
    wb_df = wb_df.merge(temp_data['temperature(C)'].resample(resolution).min(), how='left', left_index=True,
                        right_index=True)
    wb_df.rename(columns={'temperature(C)': 'min_temperature(C)'}, inplace=True)
    wb_df['valid'] = np.where(wb_df['min_temperature(C)'].isna(), False, wb_df['valid'])

    wb_df['min_temperature(C)'].ffill().bfill(inplace=True)

    # add a column with max temperature
    wb_df = wb_df.merge(temp_data['temperature(C)'].resample(resolution).max(), how='left', left_index=True,
                        right_index=True)
    wb_df.rename(columns={'temperature(C)': 'max_temperature(C)'}, inplace=True)
    wb_df['max_temperature(C)'].ffill().bfill(inplace=True)

    # add a column with mean temperature
    wb_df['mean_temperature(C)'] = wb_df[['min_temperature(C)', 'max_temperature(C)']].mean(axis=1)

    return wb_df, temp_data


def create_rain_and_snow_input(wb_df, temp_data, meteo_name, variable_parameters, path_to_data_folder, resolution):
    # load precipitation data
    precip_data = import_data_from_csv_file(find_file_by_name(f'{meteo_name}_precip_H',
                                                              path_to_data_folder, 'csv'))
    precip_data.rename(columns={'rre150h0': 'precipitation(mm)'}, inplace=True)

    # add a column with precipitation
    wb_df = wb_df.merge(precip_data['precipitation(mm)'].resample(resolution).sum(), how='left', left_index=True, right_index=True)

    # exclude (set valid to False) data gaps from the calibration period
    wb_df['valid'] = np.where(wb_df['precipitation(mm)'].isna(), False, wb_df['valid'])
    #wb_df['precipitation(mm)'].fillna(0, inplace=True)

    meteo_data = precip_data.merge(temp_data['temperature(C)'], how='inner', left_index=True, right_index=True)

    # compute rain fall and snow fall with hourly resolution depending on snow melt temperature
    melt_rate = variable_parameters['melting_rate']
    melt_temp = variable_parameters['melting_temperature']
    meteo_data['rain_fall(mm)'] = np.where(meteo_data['temperature(C)'] > melt_temp, meteo_data['precipitation(mm)'], 0)
    meteo_data['snow_fall(mm)'] = np.where(meteo_data['temperature(C)'] < melt_temp, meteo_data['precipitation(mm)'], 0)

    # aggregate rain and snow fall data
    wb_df['rain_fall(mm)'] = meteo_data['rain_fall(mm)'].resample(resolution).sum()  # [mm water equivalent]
    wb_df['snow_fall(mm)'] = meteo_data['snow_fall(mm)'].resample(resolution).sum()  # [mm water equivalent]

    # set columns initially to zero
    num_rows = len(wb_df)  # Get the number of rows from the existing DataFrame
    wb_df['snow_melt(mm)'] = [0.0] * num_rows  # snow melt in mm water equivalent
    wb_df['snow_cover(mm)'] = [0.0] * num_rows  # snow cover in mm water equivalent

    # iterate over df row by row to compute snow cover and snow melt
    y = wb_df.index.tolist()[0]
    for i in wb_df.index.tolist():
        # Compute Snow melt
        if wb_df['mean_temperature(C)'][i] > melt_temp:
            wb_df.loc[i, 'snow_melt(mm)'] = np.minimum(melt_rate * (wb_df['mean_temperature(C)'][i] - melt_temp),
                                                       wb_df['snow_cover(mm)'][y])

        # Compute Snow cover
        wb_df.loc[i, 'snow_cover(mm)'] = (wb_df['snow_cover(mm)'][y] - wb_df['snow_melt(mm)'][i] +
                                          wb_df['snow_fall(mm)'][i])
        y = i  # current day is new yesterday

    return wb_df


def create_potential_evapotranspiration_input(wb_df, resolution):
    num_rows = len(wb_df)  # Get the number of rows from the existing DataFrame
    wb_df['pET(mm)'] = [0] * num_rows  # potential evapotranspiration

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


def create_model_input_df(spring_name, meteo_name, fixed_parameters, variable_parameters, path_to_data_folder, optimize=False, resolution='D'):
    # create a dataframe to store the water balance model timeseries
    wb_df = initialize_wb_df(spring_name, path_to_data_folder, optimize)

    # adds a column with the measured spring_name discharge
    wb_df = create_spring_input(spring_name, path_to_data_folder, variable_parameters, resolution)

    # adds column with min, max, mean temperature
    wb_df, temp_data = create_temperature_input(wb_df, meteo_name, path_to_data_folder, resolution)

    # adds column with precipitation, rain fall, snow fall, snow cover, snow melt
    wb_df = create_rain_and_snow_input(wb_df, temp_data, meteo_name, variable_parameters, path_to_data_folder, resolution)

    wb_df = create_potential_evapotranspiration_input(wb_df, resolution)

    # adds empty columns for the model calculations
    wb_df = create_model_parameter_columns(wb_df)

    return wb_df


def initialize_model(spring_name, path_to_data_folder):
    if spring_name == 'Ulrika':
        meteo_name = 'Freienbach'
    elif spring_name == 'Paliu_Fravi':
        meteo_name = 'Chur'
    else:
        raise ValueError(f"expected 'Ulrika' or 'Paliu_Fravi' as spring_name but got {spring_name}.")

    fixed_parameters = get_catchment_parameters(spring_name)
    variable_parameters, pet_Hamon_parameters = get_model_parameters()
    wb_df = create_model_input_df(spring_name, meteo_name, fixed_parameters, variable_parameters, path_to_data_folder)

    return fixed_parameters, variable_parameters, wb_df


def find_file_by_name(filename, startFolder, filetype):
    # search for the filepath of a single file
    if filetype.lower() not in filename.lower():
        filename = '{}.{}'.format(filename, filetype.lower())  # add filetype

    filefound = False
    for root, dirs, files in os.walk(startFolder):  # Walking top-down from the startFolder looking for the file
        if filename.lower() in [file.lower() for file in files]:
            filefound = True
            path_to_file = os.path.join(root, filename)

    if not filefound:
        path_to_file = ''
        print('{} not found within directory {}'.format(filename, startFolder))
    return path_to_file


def import_data_from_csv_file(filepath):
    df = pd.read_csv(filepath)  # read csv
    df['datetime'] = pd.to_datetime(df['datetime'])
    # convert column to datetime format
    df.set_index('datetime', inplace=True)  # set date as index
    return df
