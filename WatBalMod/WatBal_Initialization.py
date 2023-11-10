import pandas as pd  # data processing
import numpy as np  # data processing

import Data_Import
import Helper


def get_model_parameters():
    # Catchment properties
    catchment_parameters = {
        'area': 389459,  # estimated area [m ^ 2]
        'latitude': 46.5,  # mean Latitude [deg]
        'elevation': 1590  # mean elevation [masl]
    }

    # Model parameters
    model_parameters = {
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

    return catchment_parameters, model_parameters, pet_Hamon_parameters


def create_spring_input(spring_name, path_to_data_folder, catchment_parameters, resolution='D'):
    # load spring data
    spring_data = Data_Import.import_data_from_csv_file(Helper.find_file_by_name(f'{spring_name}_{resolution}',
                                                                                 path_to_data_folder, 'csv'))

    # Create a DataFrame to store model variables
    wb_df = pd.DataFrame(index=spring_data.index)
    wb_df['doy'] = wb_df.index.dayofyear  # Julien day = day of the year

    wb_df['discharge_meas(mm)'] = spring_data['discharge(L/min)'] * (60 * 24) / catchment_parameters['area']
    return wb_df


def create_temperature_input(wb_df, meteo_name, path_to_data_folder, resolution):

    # load temperature data
    temp_data = Data_Import.import_data_from_csv_file(Helper.find_file_by_name(f'{meteo_name}_temp_H',
                                                                               path_to_data_folder, 'csv'))

    # add a column with min temperature
    wb_df = wb_df.merge(temp_data['temperature(C)'].resample(resolution).min(), how='left', left_index=True,
                        right_index=True)
    wb_df.rename(columns={'temperature(C)': 'min_temperature(C)'}, inplace=True)

    wb_df['min_temperature(C)'].ffill().bfill(inplace=True)

    # add a column with max temperature
    wb_df = wb_df.merge(temp_data['temperature(C)'].resample(resolution).max(), how='left',left_index=True,
                        right_index=True)
    wb_df.rename(columns={'temperature(C)': 'max_temperature(C)'}, inplace=True)
    wb_df['max_temperature(C)'].ffill().bfill(inplace=True)

    # add a column with mean temperature
    wb_df['mean_temperature(C)'] = wb_df[['min_temperature(C)', 'max_temperature(C)']].mean(axis=1)

    return wb_df, temp_data


def create_rain_and_snow_input(wb_df, temp_data, meteo_name, model_parameters, path_to_data_folder, resolution):
    # load precipitation data
    precip_data = Data_Import.import_data_from_csv_file(Helper.find_file_by_name(f'{meteo_name}_precip_H',
                                                                                 path_to_data_folder, 'csv'))
    precip_data.rename(columns={'rre150h0': 'precipitation(mm)'}, inplace=True)

    # add a column with precipitation
    wb_df = wb_df.merge(precip_data['precipitation(mm)'].resample(resolution).sum(), how='left', left_index=True, right_index=True)
    wb_df['precipitation(mm)'].fillna(0, inplace=True)

    meteo_data = precip_data.merge(temp_data['temperature(C)'], how='inner', left_index=True, right_index=True)

    # compute rain fall and snow fall with hourly resolution depending on snow melt temperature
    melt_rate = model_parameters['melting_rate']
    melt_temp = model_parameters['melting_temperature']
    meteo_data['rain_fall(mm)'] = np.where(meteo_data['temperature(C)'] > melt_temp, meteo_data['precipitation(mm)'], 0)
    meteo_data['snow_fall(mm)'] = np.where(meteo_data['temperature(C)'] < melt_temp, meteo_data['precipitation(mm)'], 0)

    # aggregate rain and snow fall data
    wb_df['rain_fall(mm)'] = meteo_data['rain_fall(mm)'].resample(resolution).sum()  # [mm water equivalent]
    wb_df['snow_fall(mm)'] = meteo_data['snow_fall(mm)'].resample(resolution).sum()  # [mm water equivalent]

    '''
    # Compute Snow cover and snow melt in mm water equivalent
    wb_df['snow_fall_sum(mm)'] = wb_df['snow_fall(mm)'].cumsum()
    wb_df['snow_melt_pot(mm)'] = np.maximum((wb_df['mean_temperature(C)'] - melt_temp) * melt_rate, 0)
    wb_df['snow_melt_pot_sum(mm)'] = wb_df['snow_melt_pot(mm)'].cumsum()
    wb_df['snow_cover(mm)'] = np.maximum(wb_df['snow_fall_sum(mm)'] - wb_df['snow_melt_pot_sum(mm)'], 0)
    wb_df['snow_melt(mm)'] = np.where(wb_df['snow_melt_pot(mm)'] < wb_df['snow_cover(mm)'], wb_df['snow_melt_pot(mm)'], wb_df['snow_cover(mm)'])

    
    for i in range(1, len(wb_df)):
        # Compute Snow melt
        wb_df.at[i, 'snow_melt(mm)'] = np.minimum(melt_rate * (wb_df.at[i, 'mean_temperature(C)'] - melt_temp),
                                                  wb_df.at[i - 1, 'snow_cover(mm)'])

        # Compute Snow cover (SC)
        wb_df.at[i, 'snow_cover(mm)'] = (wb_df.at[i - 1, 'snow_cover(mm)'] - wb_df.at[i, 'snow_melt(mm)'] +
                                         wb_df.at[i, 'snow_fall(mm)'])
    
    wb_df['snow_fall_sum(mm)'] = wb_df['snow_fall(mm)'].cumsum()
    wb_df['snow_cover(mm)'] = wb_df['snow_cover(mm)'].shift(1) - wb_df['snow_melt(mm)'] + wb_df['snow_fall(mm)']

    # Compute Snow melt
    wb_df['snow_melt(mm)'] = np.minimum(k * (wb_df['T_b'] - Tsm), wb_df['snow_cover(mm)'].shift(1))

    # Compute Snow cover (SC)
    wb_df['snow_cover(mm)'] = wb_df['snow_cover(mm)'].shift(1) - wb_df['snow_melt(mm)'] + wb_df['snow_fall(mm)']
    '''

    return wb_df


def create_potential_evapotranspiration_input(wb_df, resolution):
    num_rows = len(wb_df)  # Get the number of rows from the existing DataFrame
    wb_df['pET(mm)'] = [0] * num_rows  # potential evapotranspiration

    return wb_df

def create_model_parameter_columns(wb_df):
    num_rows = len(wb_df)  # Get the number of rows from the existing DataFrame
    # Add columns for model variables with initial values of 0
    wb_df['aET(mm)'] = [0] * num_rows  # actual evapotranspiration
    wb_df['snow_melt(mm)'] = [0] * num_rows  # snow melt in mm water equivalent
    wb_df['snow_cover(mm)'] = [0] * num_rows  # snow cover in mm water equivalent
    wb_df['storage_soil(mm)'] = [0] * num_rows  # water saturation in the soil reservoir
    wb_df['storage_gw(mm)'] = [0] * num_rows  # water storage in the groundwater reservoir
    wb_df['runoff(mm)'] = [0] * num_rows  # surface runoff
    wb_df['percolation_gw(mm)'] = [0] * num_rows  # percolation from saturated soil to groundwater reservoir
    wb_df['discharge_sim(mm)'] = [0] * num_rows  # spring discharge simulated

    return wb_df


def create_model_input_df(spring_name, meteo_name, catchment_parameters, model_parameters, path_to_data_folder, resolution='D'):
    # create a dataframe to store the water balance model timeseries
    # adds a column with the measured spring discharge
    wb_df = create_spring_input(spring_name, path_to_data_folder, catchment_parameters, resolution)

    # adds column with min, max, mean temperature
    wb_df, temp_data = create_temperature_input(wb_df, meteo_name, path_to_data_folder, resolution)

    # adds column with precipitation, rain fall, snow fall, snow cover, snow melt
    wb_df = create_rain_and_snow_input(wb_df, temp_data, meteo_name, model_parameters, path_to_data_folder, resolution)

    wb_df = create_potential_evapotranspiration_input(wb_df, resolution)

    # adds empty columns for the model calculations
    wb_df = create_model_parameter_columns(wb_df)

    return wb_df


def create_input_dataframe(spring_name, meteo_name, path_to_data_folder, catchment_parameters, resolution='D'):
    # currently only daily resolution
    # load spring data
    spring_data = Data_Import.import_data_from_csv_file(Helper.find_file_by_name(f'{spring_name}_{resolution}',
                                                                                 path_to_data_folder, 'csv'))

    # load precipitation data
    precip_data = Data_Import.import_data_from_csv_file(Helper.find_file_by_name(f'{meteo_name}_precip_H',
                                                                                 path_to_data_folder, 'csv'))

    # load temperature data
    temp_data = Data_Import.import_data_from_csv_file(Helper.find_file_by_name(f'{meteo_name}_temp_H',
                                                                               path_to_data_folder, 'csv'))

    # Create a DataFrame to store model variables
    input_df = pd.DataFrame(index=spring_data.index)
    input_df['doy'] = input_df.index.dayofyear  # Julien day = day of the year

    input_df['discharge_meas(mm)'] = spring_data['discharge(L/min)'] * (60 * 24) / catchment_parameters['area']

    input_df = input_df.merge(precip_data['rre150h0'], how='left', left_index=True,
                                                   right_index=True)
    input_df.rename(columns={'rre150h0': 'precipitation(mm)'}, inplace=True)
    input_df['precipitation(mm)'].fillna(0, inplace=True)

    input_df = input_df.merge(temp_data['temperature(C)'].resample(resolution).min(), how='left',
                                                    left_index=True, right_index=True)
    input_df.rename(columns={'temperature(C)': 'min_temperature(C)'}, inplace=True)

    input_df['min_temperature(C)'].ffill().bfill(inplace=True)

    input_df = input_df.merge(temp_data['temperature(C)'].resample(resolution).max(), how='left',
                                                    left_index=True, right_index=True)
    input_df.rename(columns={'temperature(C)': 'max_temperature(C)'}, inplace=True)
    input_df['max_temperature(C)'].ffill().bfill(inplace=True)

    input_df['mean_temperature(C)'] = input_df[['min_temperature(C)', 'max_temperature(C)']].mean(axis=1)

    num_rows = len(input_df)  # Get the number of rows from the existing DataFrame
    # Add columns for model variables with initial values of 0
    input_df['saturation(mm)'] = [0] * num_rows  # water saturation in the soil reservoir
    input_df['storage_gw(mm)'] = [0] * num_rows  # water storage in the groundwater reservoir
    input_df['snow_cover(mm)'] = [0] * num_rows  # snow cover in mm water equivalent
    input_df['rain_fall(mm)'] = [0] * num_rows  # rain
    input_df['snow_fall(mm)'] = [0] * num_rows  # snow fall in mm water equivalent
    input_df['snow_melt(mm)'] = [0] * num_rows  # snow melt in mm water equivalent
    input_df['runoff(mm)'] = [0] * num_rows  # surface runoff
    input_df['pET(mm)'] = [0] * num_rows  # potential evapotranspiration
    input_df['aET(mm)'] = [0] * num_rows  # actual evapotranspiration
    input_df['percolation_gw(mm)'] = [0] * num_rows  # percolation from saturated soil to groundwater reservoir
    input_df['discharge_sim(mm)'] = [0] * num_rows  # spring discharge simulated

    return input_df
