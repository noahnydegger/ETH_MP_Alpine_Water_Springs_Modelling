from Water_Balance_Model import wb_Data_preparation
from Water_Balance_Model import wb_Initialization
from Data import datafolder

spring_name = 'Ulrika'
meteo_names = ['Freienbach', 'Oberriet_Kriessern']


wb_discharge = wb_Data_preparation.create_model_input_discharge(spring_name, datafolder)

wb_precip = wb_Data_preparation.create_model_input_precipitation(spring_name, meteo_names, datafolder)

elevation_difference = 647-415  # [m] elevation difference between meteo stations relative to first station
wb_temp = wb_Data_preparation.create_model_input_temperature(spring_name, meteo_names, elevation_difference, datafolder)

wb_df = wb_Initialization.initialize_wb_df(spring_name, datafolder, False)
wb_df.to_csv('wb_df_init.csv', index=True)
