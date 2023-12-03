from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import Data


srping_name = 'Ulrika'
stations = ['Freienbach', 'Oberriet_Kriessern']

path_to_precip_pkl = Data.datafolder / 'meteo_data' / 'resampled_precip_data' / 'resampled_precip_data_dfs.pkl'
with open(path_to_precip_pkl, 'rb') as file:
    resampled_precip_data_dfs = pickle.load(file)

merged_precip = pd.merge(resampled_precip_data_dfs[stations[0]]['D'], resampled_precip_data_dfs[stations[1]]['D'], how='left', left_index=True,
                        right_index=True)

merged_precip.columns = stations

# Create a scatter plot
plt.scatter(merged_precip[stations[0]], merged_precip[stations[1]], color='blue', )

# Add labels and title
plt.xlabel(stations[0])
plt.ylabel(stations[1])
plt.title('Scatter Plot of Precipitation Data')

# Show the plot
plt.show()

# Extract the features and target
X = merged_precip[stations[0]]
y = merged_precip[stations[1]]

# Fit a polynomial of degree 3
coefficients = np.polyfit(X, y, 3)

# Print the coefficients
print('Coefficients:', coefficients)

# Generate points for the fitted curve
X_fit = np.linspace(min(X), max(X), 100)
y_fit = np.polyval(coefficients, X_fit)

# Plot the original data and the fitted curve
plt.scatter(X, y, color='black', label='Original Data')
plt.plot(X_fit, y_fit, color='blue', linewidth=3, label='Polynomial Fit (Degree 3)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression (Degree 3)')
plt.legend()
plt.show()

merged_precip['precip_filled'] = merged_precip[stations[0]].fillna(merged_precip[stations[1]])
