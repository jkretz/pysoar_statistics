import os
import pandas as pd
import re
import glob
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json


def main():

    # Change path to pysoar output
    ipath_data = './pysoar_bin'

    opath_plots = './plots'
    data = pysoar_to_dict(ipath_data)

    # Read data from file:
    pilot_cn = 'JK'
    selection = json.load(open(f'plots/{pilot_cn}/comp_selection.json'))

    opath_pilot = os.path.join(opath_plots, pilot_cn)
    if not os.path.exists(opath_pilot):
        os.makedirs(opath_pilot)

    select_competitors = process_competitors(selection['comps'], data)

    ifile_reg = 'regression_coeff/save_regs_std.p'
    speed_abs, speed_abs_pred, speed_var_pred = competitor_analysis(select_competitors, ifile_reg)

    speed_abs, speed_abs_pred, speed_var_pred, speed_var_pred_avg = (
        data_postprocess(speed_abs, speed_abs_pred, speed_var_pred))

    # Plot speed predictions
    plot_speed_predictions(speed_abs['all'], speed_abs_pred['all'], pilot_cn, opath_pilot)

    # Plot histograms
    plot_histograms(speed_var_pred['all'], pilot_cn, opath_pilot)

    plot_year_averages(speed_var_pred_avg, pilot_cn, opath_pilot)


def plot_year_averages(speed_var_pred_avg, pilot_name, opath):
    data_var_list = None
    for _, data in speed_var_pred_avg.items():
        data_var_list = list(data.keys())
        break
    year_list = sorted(speed_var_pred_avg.keys())

    figure, ax = plt.subplots()
    for var in data_var_list:
        data_var = []
        for year in year_list:
            data_var.append(speed_var_pred_avg[year][var])
        ax.plot(pd.to_datetime(year_list), data_var, label=var)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_ylabel(r'Predicted $\Delta$Task Speed (km/h)')
    ax.set_xlabel('Year')
    ax.set_ylim(-8, 6)
    ax.axhline(0, color='black', linestyle='--')

    plt.legend()
    plt.savefig(f'{opath}/{pilot_name}_evolution.pdf')
    plt.close()


def plot_speed_predictions(speed_abs, speed_abs_pred, pilot_name, opath):
    """Plot the predicted vs. actual speed."""
    plt.plot(sorted(speed_abs), sorted(speed_abs))
    plt.scatter(speed_abs, speed_abs_pred)
    plt.xlabel(r'$\Delta$Task Speed Top 3 (km/h)')
    plt.ylabel(r'Predicted $\Delta$Task Speed Top 3 (km/h)')
    plt.savefig(f'{opath}/{pilot_name}_pred.pdf')
    plt.close()


def plot_histograms(speed_var_pred, pilot_name, opath):
    """Plot histograms for each parameter."""
    fig, ax = plt.subplots(2, 2, sharey=True, figsize=(10, 10))
    ax = ax.flatten()

    for nv, var in enumerate(['Average cruise speed (GS)', 'Average rate of climb', 'Average L/D',
                              'Excess distance covered']):
        # Plot histogram for each metric
        ax[nv].set_xlabel(r'Predicted $\Delta$Task Speed (km/h)')
        ax[nv].set_ylabel('Frequency')
        ax[nv].set_title(var)
        ax[nv].set_xlim(-14, 14)
        ax[nv].hist(speed_var_pred[var], bins=np.arange(-14, 15, 2),
                    label=f'Mean: {np.nanmean(speed_var_pred[var]):.2f} km/h')
        # ax[nv].hist(speed_var_pred[var], bins=np.arange(-14, 15, 2),
        #             label=f'Median: {np.nanmedian(speed_var_pred[var]):.2f} km/h')
        ax[nv].axvline(0, color='black')
        ax[nv].legend(loc='upper left', frameon=False)

    plt.tight_layout()
    plt.savefig(f'{opath}/{pilot_name}_hist.pdf')
    plt.close()


def data_postprocess(speed_abs, speed_abs_pred, speed_var_pred):
    speed_abs['all'], speed_abs_pred['all'] = [], []
    for ny, year in enumerate(speed_abs.keys()):
        if year != 'all':
            speed_abs['all'] += speed_abs[year]
            speed_abs_pred['all'] += speed_abs_pred[year]

    list_data = list(speed_var_pred.values())
    list_year = list(speed_var_pred.keys())

    speed_var_pred['all'] = {}
    speed_var_pred_avg = {}
    for ny, data_year in enumerate(zip(list_data, list_year)):
        speed_var_pred_avg[data_year[1]] = {}
        for var, data_var in data_year[0].items():
            if ny == 0:
                speed_var_pred['all'][var] = data_var
            else:
                speed_var_pred['all'][var] += data_var
            speed_var_pred_avg[data_year[1]][var] = np.nanmean(data_var)
            # speed_var_pred_avg[data_year[1]][var] = np.nanmedian(data_var)

    return speed_abs, speed_abs_pred, speed_var_pred, speed_var_pred_avg


def competitor_analysis(select_competitors, ifile_reg):
    val_reg, speed_bins = load_data(ifile_reg)

    data_pilot_best_abs, data_pilot_best_rel, data_pilot_best = process_pilot_data(select_competitors)

    speed_abs, speed_abs_pred, speed_var_pred = {}, {}, {}
    for year in data_pilot_best_abs.keys():
        speed_indices = precompute_speed_indices(speed_bins, data_pilot_best[year], data_pilot_best_abs[year])

        speed_abs[year], speed_abs_pred[year] = (
            calculate_predicted_speed(data_pilot_best[year], data_pilot_best_abs[year], val_reg, speed_indices))

        speed_var_pred[year] = calculate_predicted_speed_variabel(
            data_pilot_best[year], data_pilot_best_abs[year], val_reg, speed_indices)

    return speed_abs, speed_abs_pred, speed_var_pred


def calculate_predicted_speed_variabel(data_pilot_best, data_pilot_best_abs, val_reg, speed_indices):
    delta_task_speed = {}

    for nv, val in enumerate(['Average cruise speed (GS)', 'Average rate of climb', 'Average L/D',
                              'Excess distance covered']):
        delta_task_speed[val] = []

        for data in zip(data_pilot_best['Task speed'], data_pilot_best_abs[val]):
            if any(np.isnan(data)):
                continue
            ind_speed = speed_indices.get(data[0], None)  # Retrieve precomputed index
            if ind_speed is None:
                continue

            delta_task_speed[val].append(val_reg[val][ind_speed] * data[1])

    return delta_task_speed


def calculate_predicted_speed(data_pilot_best, data_pilot_best_abs, val_reg, speed_indices):
    """Calculate the predicted speeds for each pilot day."""
    speed_abs, speed_abs_pred, drop_data = [], [], []
    for nd, (speed, speed_abs_data) in enumerate(zip(data_pilot_best['Task speed'], data_pilot_best_abs['Task speed'])):
        if np.isnan(speed):
            continue
        speed_abs.append(speed_abs_data)

        # Predict speed for the day
        ind_speed = speed_indices.get(speed, None)  # Retrieve precomputed index
        if ind_speed is None:
            continue  # Skip if no valid index found

        speed_pred_day = sum(val_reg[val][ind_speed] * data_pilot_best_abs[val][nd] for val in [
            'Average cruise speed (GS)', 'Average rate of climb', 'Average L/D', 'Excess distance covered'])

        speed_abs_pred.append(speed_pred_day)

    return speed_abs, speed_abs_pred


def process_pilot_data(select_competitors):
    """Process the pilot data, applying the necessary conditions and filling the data structures."""

    data_pilot_best_abs, data_pilot_best_rel, data_pilot_best = initialize_data_structures(select_competitors)
    year_list = list(select_competitors['data_pilot'].keys())

    for year in year_list:
        for day_all in select_competitors['data_pilot'][year].keys():

            if len(select_competitors['data_pilot'][year][day_all]) == 0:
                continue
            task_speed_best = select_competitors['data_best'][year][day_all]['Task speed']
            task_speed_pilot = select_competitors['data_pilot'][year][day_all]['Task speed']

            # Vectorized computation for the relative speed condition
            rel_condition = np.abs((task_speed_pilot / task_speed_best) - 1) < 0.3
            for var in data_pilot_best_abs[year].keys():
                if rel_condition:
                    val_best = select_competitors['data_best'][year][day_all][var]
                    val_pilot = select_competitors['data_pilot'][year][day_all][var]

                    val_rel = val_pilot / val_best
                    val_abs = val_pilot - val_best

                    # If the relative value condition holds, append the data
                    if np.abs(val_rel - 1) < 1.0:
                        data_pilot_best_abs[year][var].append(val_abs)
                        data_pilot_best_rel[year][var].append(val_rel - 1)
                        data_pilot_best[year][var].append(val_best)
                    else:
                        data_pilot_best_abs[year][var].append(np.nan)
                        data_pilot_best_rel[year][var].append(np.nan)
                        data_pilot_best[year][var].append(np.nan)
                else:
                    # If the speed condition does not hold, append NaNs
                    data_pilot_best_abs[year][var].append(np.nan)
                    data_pilot_best_rel[year][var].append(np.nan)
                    data_pilot_best[year][var].append(np.nan)
    return data_pilot_best_abs, data_pilot_best_rel, data_pilot_best


def precompute_speed_indices(speed_bins, data_pilot_best, data_pilot_best_abs):
    """Precompute the speed indices for faster access."""
    unique_speeds = np.unique(np.concatenate([data_pilot_best['Task speed'], data_pilot_best_abs['Task speed']]))
    speed_indices = {speed: np.argmin(np.abs(speed_bins - speed)) for speed in unique_speeds}
    return speed_indices


def initialize_data_structures(select_competitors):
    """Initialize the dictionaries to store pilot data."""

    # Get the first available year's data dictionary (if exists)
    first_year = next(iter(select_competitors['data_pilot'].values()), None)

    # Get the first available day's data dictionary (if exists)
    first_day = next(iter(first_year.values()), None) if first_year else None

    # Extract keys if data exists
    list_keys = list(first_day.keys()) if first_day else []

    # Ensure each year gets a separate dictionary copy
    data_pilot_best_abs = {year: {key: [] for key in list_keys} for year in select_competitors['data_best'].keys()}
    data_pilot_best_rel = {year: {key: [] for key in list_keys} for year in select_competitors['data_best'].keys()}
    data_pilot_best = {year: {key: [] for key in list_keys} for year in select_competitors['data_best'].keys()}

    return data_pilot_best_abs, data_pilot_best_rel, data_pilot_best


def load_data(ifile):
    """Load the regression model and speed bins."""
    val_reg = pickle.load(open(ifile, 'rb'))
    speed_bins = val_reg.pop('speed_bins')
    return val_reg, speed_bins


def pysoar_to_dict(ipath):
    data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # Efficiently list .xls files
    xls_files = list(glob.iglob(os.path.join(ipath, "**", "*.xls"), recursive=True))

    # Parallelize file processing
    with ThreadPoolExecutor(max_workers=max(os.cpu_count() - 2, 1)) as executor:
        results = executor.map(process_file, xls_files)

    # Populate dictionary
    for file_path, comp_date, comp_year, comp_class, comp_name, data in results:
        data_dict[comp_year][comp_name][comp_class][comp_date] = data

    return data_dict


def process_file(file_path):
    """Helper function for parallel processing."""
    comp_date, comp_year, comp_class, comp_name = competition_info(file_path)
    data = import_day_xls(file_path)
    return file_path, comp_date, comp_year, comp_class, comp_name, data


def import_day_xls(ifile):
    df = pd.read_excel(ifile, header=1, skiprows=[2, 3])
    df = df.drop(columns=['Airplane'], errors='ignore')  # Avoids KeyError
    return df.to_dict()


def competition_info(ipath):
    ipath_in_split = ipath.split(os.path.sep)  # Ensures cross-platform compatibility
    if len(ipath_in_split) < 4:
        raise ValueError("File path does not contain enough hierarchy levels.")

    date_out = ipath_in_split[-2]
    if not re.match(r'.*-.*-', date_out):  # Simplified regex check
        raise ValueError('Second-to-last entry is not a date')

    year_out = date_out.split('-')[-1]
    comp_class_out = ipath_in_split[-3]
    comp_out = ipath_in_split[-4]

    return date_out, year_out, comp_class_out, comp_out


def process_competitors(selection_in, data_in):
    """
    Processes selected competitors and organizes data into dictionaries.

    Parameters:
    - selection_in: List of selected competition details.
    - data_in: Dictionary containing competition data.

    Returns:
    - Dictionaries for pilot, first, best, and all competitors.
    """

    # Get unique years from selection_in
    year_all = {comp[0] for comp in selection_in}

    # Initialize dictionaries using a dictionary comprehension
    data_pilot = {year: {} for year in year_all}
    data_first = {year: {} for year in year_all}
    data_best = {year: {} for year in year_all}
    data_all = {year: {} for year in year_all}

    for year, comp_name, comp_class, comp_id in selection_in:
        data_comp = data_in[year][comp_name][comp_class]

        for day, data_day in data_comp.items():
            key = f"{day}{comp_class}"

            # Store results in dictionaries
            selected_pilot = select_competitor(data_day, comp_id, comp_id)
            selected_first = select_competitor(data_day, "first", comp_id)
            selected_best = select_competitor(data_day, "best", comp_id)

            data_pilot[year][key] = selected_pilot
            data_first[year][key] = selected_first
            data_best[year][key] = selected_best

            # Store all competitors
            data_all[year][key] = {comp: select_competitor(data_day, comp, comp)
                                   for comp in data_day['Callsign'].values()}

    return {'data_pilot': data_pilot, 'data_first': data_first, 'data_best': data_best, 'data_all': data_all}


def select_competitor(data_day, competitor, comp_num):
    """
    Selects the appropriate competitor(s) from the dataset.

    Parameters:
    - data_day: Competition data for a specific day.
    - competitor: Specific competitor to select.
    - comp_num: Competitor number.

    Returns:
    - Dictionary with selected competitor's data.
    """
    data_out = {}
    unused_keys = {'Ranking', 'Callsign', 'Start time', 'Finish time',
                   'Height loss during circling', 'Average thermal speed (GS)'}  # O(1) lookup

    if competitor == comp_num:
        for rank, callsign in data_day['Callsign'].items():
            if competitor == callsign:
                for key in data_day.keys():
                    if key not in unused_keys:
                        data_out[key] = compute_competitor_data(data_day, key, rank=rank, avg=False)
        return data_out

    elif competitor == "first":
        return {key: compute_competitor_data(data_day, key, rank=0, avg=False)
                for key in data_day.keys() if key not in unused_keys}

    elif competitor == "best":
        return {key: compute_competitor_data(data_day, key, rank="best", avg=True)
                for key in data_day.keys() if key not in unused_keys}

    return data_out


def compute_competitor_data(data_in, key_in, **kwargs):
    """
    Computes either a single competitor's metric or an average over multiple competitors.

    Parameters:
    - data_in: Input competition data.
    - key_in: Key of the metric to process.

    Returns:
    - Computed value for the given key.
    """
    best_len = 3  # Number of top competitors to consider for "best"

    if not kwargs['avg']:  # Case: Single competitor
        return data_in[key_in][kwargs['rank']]

    elif kwargs['rank'] in {"all", "best"}:  # Case: Compute average
        rank_list = np.arange(len(data_in['Ranking'])) if kwargs['rank'] == "all" else np.arange(best_len)

        # Use NumPy vectorization for efficiency
        avg_values = np.array([data_in[key_in][rank] for rank in rank_list if not np.isnan(data_in[key_in][rank])])

        return np.nanmean(avg_values) if avg_values.size > 0 else np.nan
        # return np.nanmedian(avg_values) if avg_values.size > 0 else np.nan

    else:
        raise ValueError("Cannot compute an average for a single competitor.")


def normalize_pilot(data_pilot, data_best):
    """
    Normalizes the pilot's performance relative to the best performance.

    Returns:
    - abs_best: Absolute difference from the best performance.
    - rel_best: Relative difference from the best performance.
    """
    abs_best = {key: data_pilot[key] - data_best[key] for key in data_pilot}
    rel_best = {key: abs_best[key] / data_best[key] for key in data_pilot if data_best[key] != 0}

    return abs_best, rel_best


if __name__ == '__main__':
    main()
