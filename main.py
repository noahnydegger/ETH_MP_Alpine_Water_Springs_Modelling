import os

import Data_Import
import Data_Cleaning
import Data_Visualization

path_to_data_folder = 'Data'
latest_data_directory = 'wabesense_discharge_2023-09-01'

path_to_plot_folder = 'Plots'
plotFileType = 'pdf'  # 'pdf' or 'png'; filetype for the plots

if not os.path.isdir(path_to_plot_folder):  # creates the folder if it does not exist yet
    os.makedirs(path_to_plot_folder)


if __name__ == '__main__':
    # load the spring data into a list of dataframes
    spring_names, spring_description, spring_data_paths, spring_data_dfs = Data_Import.import_spring_data(
        os.path.join(path_to_data_folder, latest_data_directory))

    fig_all_springs_go = Data_Visualization.plot_interactive_figure(spring_data_dfs, spring_names)
    fig_all_springs_go.show()
    for i in range(0, len(spring_data_dfs)):
        fig_single_spring = Data_Visualization.plot_single_spring(spring_data_dfs[i], spring_names[i])
        fig_single_spring.savefig(os.path.join(path_to_plot_folder, '{}.{}'.format(spring_names[i], plotFileType)))
