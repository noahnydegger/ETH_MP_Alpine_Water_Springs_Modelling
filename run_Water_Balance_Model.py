from pathlib import Path
from scipy import optimize
import Data
from Water_Balance_Model import wb_Initialization
from Water_Balance_Model import wb_Computation
from Water_Balance_Model import wb_Visualization
import time

# specify the spring
spring_name = 'Ulrika'
optimize_model = False

# create the model input if is not available yet


# initialize the model
fixed_parameters, variable_parameters, wb_df = wb_Initialization.initialize_model(spring_name, Data.datafolder)

if optimize_model:
    # Define parameter bounds for the variable parameters
    bounds = [(estimate - 0.1 * abs(estimate), estimate + 0.1 * abs(estimate)) for estimate in variable_parameters.values()]

    # Run the global optimization
    result = optimize.differential_evolution(wb_Computation.optimize_over_full_timeseries, bounds,
                                             args=(fixed_parameters, wb_df), maxiter=2, callback=wb_Computation.callback)

    # Retrieve optimal parameters (including the fixed parameter)
    optimal_parameters = result.x
    print(optimal_parameters)
    wb_df, gof_values = wb_Computation.compute_water_balance_iterative(wb_df, fixed_parameters, optimal_parameters)
else:
    # simple run
    start_time = time.time()
    wb_df, gof_values = wb_Computation.compute_water_balance_iterative(wb_df, fixed_parameters, list(variable_parameters.values()))

    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time} seconds")

    wb_df.to_csv('wb_df_finished.csv', index=True)
# create plots
wb_Visualization.first_impression_plot(spring_name, wb_df)


