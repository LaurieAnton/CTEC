'''
Additional functions used in post-processing of results after simulations were completed.

Many functions assume multiple simulations were run, with CTEC objects stored in a dictionary.

===
Example:

inflow_df = pd.read_csv('inflow.csv')
inflow_df['Timestamp'] = pd.to_datetime(inflow_df['Timestamp'])
grouped = inflow_df.groupby(inflow_df['Timestamp'].dt.date)

result_sims = {}
for date, group_df in grouped:
    .
    .
    .
    CTECsim = CTEC.simulator(roadScenario=road)
    CTECsim.simulate()
    result_sims[date] = CTECsim
===

Here, results_sims is the dictionary.
'''

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Function to load datasets
def load_dataset(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Function to process and sum data
def process_and_sum_data(result_sims_NB, result_sims_SB, result_type='total_power', station_types=['S1', 'S2'], data_types=['LD', 'HD']):
    # Placeholder for combined datasets
    combined_data = {
        'LD_S1': [],
        'LD_S2': [],
        'HD_S1': [],
        'HD_S2': [],
        'LDHD_S1': [],
        'LDHD_S2': []
    }

    # Process and sum LD and HD data for each station and date
    for date in result_sims_NB.keys():
        if date in result_sims_SB:  # Ensure matching dates
            for station in station_types:
                for dtype in data_types:
                    nb_key = f'{station}_NB_{dtype}'
                    sb_key = f'{station}_SB_{dtype}'
                    if nb_key in result_sims_NB[date].results_dict['stationType_results'] and sb_key in result_sims_SB[date].results_dict['stationType_results']:
                        # Extracting 'total_power' data
                        nb_df = result_sims_NB[date].results_dict['stationType_results'][nb_key][result_type].copy()
                        sb_df = result_sims_SB[date].results_dict['stationType_results'][sb_key][result_type].copy()
                        
                        # Ensure alignment and sum
                        combined_df = nb_df.add(sb_df, fill_value=0)
                        
                        # Adding to the appropriate combined dataset
                        data_key = f'{dtype}_{station}'
                        combined_data[data_key].append(combined_df)

    # Now, aggregate LD and HD separately, then sum for LDHD
    for station in station_types:
        for dtype in data_types:
            data_key = f'{dtype}_{station}'
            if combined_data[data_key]:  # Check if there's data to concatenate
                combined_agg = pd.concat(combined_data[data_key]).groupby(level=0).sum()
                combined_data[data_key] = combined_agg

        # Sum LD and HD data to create LDHD, after aggregating all LD and HD data
        ld_agg = combined_data[f'LD_{station}']
        hd_agg = combined_data[f'HD_{station}']
        combined_data[f'LDHD_{station}'] = ld_agg.add(hd_agg, fill_value=0)

    return combined_data

def plot_every_day(df, result_type='total_power', width=7, height=3.5, 
                   line_color='blue', line_width=2, transparency=0.5):
    # Group the data by day and plot each group
    fig, ax = plt.subplots(figsize=(width, height))

    # Normalize the time to a specific date, for overlapping
    for (year, month, day), group in df.groupby([df.index.year, df.index.month, df.index.day]):
        # Create a new datetime index for the group with a common date (e.g., 1900-01-01)
        group.index = group.index.map(lambda dt: dt.replace(year=1900, month=1, day=1))
        # Plot with specified color, linewidth, and transparency
        ax.plot(group.index, group[result_type], color=line_color, linewidth=line_width, alpha=transparency)

    # Beautify the x-axis to show hours and minutes
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Show only the x-axis labels for the hours
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Set limits to only show one day
    ax.set_xlim([pd.Timestamp('1900-01-01 00:00:00'), pd.Timestamp('1900-01-02 00:00:00')])

    # Adding labels and title
    plt.xlabel('Time of day')
    if result_type == 'total_power':
        plt.ylabel('Total Power, kW')
    elif result_type == 'total_EVs_charging':
        plt.ylabel('Total EVs Charging')

    # Show grid
    plt.grid(True)

    # Show plot
    plt.show()












