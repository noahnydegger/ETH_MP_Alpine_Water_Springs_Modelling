from Water_Balance_Model import wb_Data_preparation
from Water_Balance_Model import wb_Initialization
from Data import datafolder

spring_name = 'Ulrika'
for spring_name in ['Ulrika', 'Paliu_Fravi']:
    if spring_name == 'Ulrika':
        meteo_names_precip = ['Freienbach', 'Oberriet_Kriessern']
        meteo_names_temp = ['Freienbach', 'Oberriet_Moos']
    elif spring_name == 'Paliu_Fravi':
        meteo_names_precip = ['Chur', 'Rothenbrunnen']
        meteo_names_temp = ['Chur', 'Thusis']

    wb_discharge = wb_Data_preparation.create_model_input_discharge(spring_name, datafolder)

    wb_precip = wb_Data_preparation.create_model_input_precipitation(spring_name, meteo_names_precip, datafolder)

    wb_temp = wb_Data_preparation.create_model_input_temperature(spring_name, meteo_names_temp, datafolder)
