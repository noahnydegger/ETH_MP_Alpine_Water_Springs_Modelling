def get_ulrika(show_plot):
    # get data of Ulrika
    spring_name = 'Ulrika'
    meteo_name = 'Freienbach'
    resolution = (
    'H', 'D')  # temporal resolution of spring_name and precipitation: '10min' 'H', 'D', 'M'  # 10min not for all meteo st.
    start = None
    end = None
    res_spring = resolution[0]
    res_precip = resolution[1]
    ulrika = resampled_spring_data_dfs[spring_name][res_spring][start:end]
    # Filter and create ulrika_d dataframe
    ulrika_d = ulrika.loc[(ulrika['discharge(L/min)'] > 0) & (ulrika['discharge(L/min)'] <= 2000)].copy()
    # Create a figure
    if show_plot:
        fig, ax_flow = plt.subplots(figsize=(15, 9))

        # Plot the spring_name data
        ax_flow.plot(ulrika_d.index, ulrika_d['discharge(L/min)'], linewidth=1, color="blue",
                 label='spring_name discharge', zorder=1)
        plt.ylabel('Discharge [l/min]')
        plt.title('Filtered Dataframe (ulrika_d)')

    return ulrika_d