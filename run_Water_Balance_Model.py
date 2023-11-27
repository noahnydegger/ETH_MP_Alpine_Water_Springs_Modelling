from pathlib import Path
import Data
from Water_Balance_Model import wb_Initialization
from Water_Balance_Model import wb_Computation
from Water_Balance_Model import wb_Visualization

# specify the spring
spring_name = 'Ulrika'

fixed_parameters, variable_parameters, wb_df = wb_Initialization.initialize_model(spring_name, Data.datafolder)
wb_df, gof_values = wb_Computation.compute_water_balance(wb_df, fixed_parameters, variable_parameters)

# create plots
wb_Visualization.first_impression_plot(spring_name, wb_df, variable_parameters)
