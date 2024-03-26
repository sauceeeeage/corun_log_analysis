import math
import re
import string

import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.metrics import r2_score
import statistics

# TODO: do the 1-2400 corun on server
# TODO: write a script to test where does the O1 to O3 optimization difference decrease to 0
opt_gap_filename = 'sawtooth_opt-gap-log'
co_run_log_filename = 'aarch64-cloud-machine_1-3000sawtooth_O3gemm_3hr_log'
'''
For cycle2 machine, L1(384KiB) should be cap at 49152 elements(8B), L2(3072KiB) should be 393216, and L3(30720KiB) should be 3932160
'''

Ki = 1024
Mi = 1024 * Ki
Gi = 1024 * Mi

def read_opt_gap_log(option):
    timestamps = []
    diffs = []
    counters = []
    Tspeedups = []
    Fspeedups = []
    O1_duration = []
    O2_duration = []
    O3_duration = []

    def parse_block(block):
        parts = block[0].strip().split(', ')
        t1 = float(parts[0].split(': ')[1][:-1])  # Extract and convert t1 value
        t2 = float(parts[1].split(': ')[1][:-1])  # Extract and convert t2 value
        t3 = float(parts[2].split(': ')[1][:-1])  # Extract and convert t3 value
        diff = float(block[1].split(': ')[1])  # Extract and convert diff value
        counter = int(block[2].split(': ')[1])  # Extract and convert counter value

        if t1 == 0:
            t1 = 1e-12  # TODO: to be determined
        if t2 == 0:
            t2 = 1e-12
        if t3 == 0:
            t3 = 1e-12
        sp1 = 1 / t1
        sp2 = 1 / t2
        sp3 = 1 / t3
        Tspeedup = (sp3 - sp1) / sp1
        # print(f"Speedup using 1/running time: {Tspeedup}")

        # Calculate the number of FLOPs for each data point
        flops1 = 2 * counter ** 3 / t1
        flops2 = 2 * counter ** 3 / t2
        flops3 = 2 * counter ** 3 / t3
        Fspeedup = (flops3 - flops1) / flops1
        # print(f"Speedup using FLOPS: {Fspeedup}")

        Tspeedups.append(Tspeedup)
        Fspeedups.append(Fspeedup)

        timestamps.append((t1, t2, t3))
        diffs.append(diff)
        counters.append(counter)
        O1_duration.append(t1)
        O2_duration.append(t2)
        O3_duration.append(t3)

    with open(opt_gap_filename, 'r') as f:
        lines = f.readlines()

    # Process each block of data
    block = []
    for line in lines:
        if line.strip():  # Check if the line is not empty
            block.append(line)
        else:
            # If an empty line is encountered, parse the block and reset it
            if block:
                # print(block)
                parse_block(block)
                block = []
    if option == 'O1':
        return counters, O1_duration
    elif option == 'O2':
        return counters, O2_duration
    elif option == 'O3':
        return counters, O3_duration
    elif option == 'Tspeedup':
        return counters, Tspeedups
    elif option == 'Fspeedup':
        return counters, Fspeedups
    else:
        print('Invalid option!')
        return None


def parsing_log():
    # read in the txt file
    with open(co_run_log_filename, 'r') as f:
        # read all lines into a string
        log_text = f.read()
        # use regex to find the lines that contain the data we want

        # Define the regular expression pattern to match the data within curly braces
        pattern = r"Log\s*{([\s\S]*?)}"

        # Find all matches of the pattern in the log text
        matches = re.findall(pattern, log_text)

        # Initialize a list to store extracted data
        extracted_data = []

        # Iterate over the matches and extract relevant information
        for match in matches:
            data = {}
            data['start'] = re.search(r"start:\s*(.*?),", match).group(1)
            data['finish'] = re.search(r"finish:\s*Some\(\s*(.*?),\s*\)", match).group(1)
            data['duration'] = re.search(r"duration:\s*Some\(\s*(.*?),\s*\)", match).group(1)
            data['prog_id'] = re.search(r"prog_id:\s*(.*?),", match).group(1)
            data['prog_name'] = re.search(r'prog_name:\s*"(.*)",', match).group(1)
            data['cmd'] = re.search(r'cmd:\s*Some\(\s*"(.*)",\s*\)', match).group(1)
            args_match = re.search(r'args:\s*Some\(\s*\[\s*([\s\S]*?)\s*pe\],\s*\)', match)
            data['args'] = [arg.strip('"') for arg in args_match.group(1).split(',')] if args_match else []
            extracted_data.append(data)

        return extracted_data


def parse_extraced_data(extracted_data):
    non_avg_x = []
    non_avg_y = []
    # Calculate the number of FLOPs for each data point
    for data in extracted_data:
        data['flops'] = None  # Initialize 'flops' with a default value
        args = data['args']
        # print(f"Args: {args}")
        for arg in args:
            # break down the args into a list of strings
            list_of_args = arg.split()
            # print(f"List of args: {list_of_args}")
            for a in list_of_args:
                # print(f"a: {a}")
                if a.startswith('"N='):
                    data['N'] = int(a.split('=')[1].strip('"'))
                    non_avg_x.append(int(a.split('=')[1].strip('"')))
                    # print(f"N: {data['N']}")
                    n_value = int(a.split('=')[1])
                    data['flops'] = n_value ** 2 * (2 * n_value - 1)
                    duration = float(data['duration'])
                    non_avg_y.append(float(data['duration']))
                    data['flops'] = data['flops'] / duration if data['flops'] is not None and duration else None

    return non_avg_x, non_avg_y


def zip_to_csv(x_val, y_val, filename):
    zipped = pd.DataFrame(zip(x_val, y_val), columns=['N', 'duration'])
    zipped.sort_values(by=['N'], inplace=True)
    zipped.reset_index(drop=True, inplace=True)
    zipped.to_csv(f'{filename}.csv', index=False)


def abnormal_points(total):
    counter = 0
    next_dur: float
    last_dur: float
    abnormal_points = []
    ideal_y_val = 1e-11
    for curr_dur in total['duration']:
        # print(f"{total.loc[counter, 'N']}, {curr_dur}")
        if curr_dur > total.loc[counter, 'N'] ** 4 * ideal_y_val:
            # print(f"curr_dur: {curr_dur}")
            abnormal_points.append((total.loc[counter, 'N'], curr_dur))
        counter += 1
    # pprint(f"len of abnormal_points: {len(abnormal_points)}")


def averaging_size_and_duration(extracted_data):
    # Create lists for x and y values for the plot
    x_values = []
    y_values = []

    x_values_uniq = {}
    for data in extracted_data:
        if data['N'] in x_values_uniq:
            old = x_values_uniq[data['N']]
            x_values_uniq[data['N']] = (old[0] + 1, old[1] + float(data['duration']))
        else:
            x_values_uniq[data['N']] = (1, float(data['duration']))

    # average the duration for each N
    curr_y_value: float
    for x_val, (num_of_runs, sum_of_duration) in x_values_uniq.items():
        curr_y_value = sum_of_duration / num_of_runs
        # if x_val in total_opt_gap['N'].tolist():
        #     curr_y_value = (curr_y_value + total_opt_gap.loc[total_opt_gap['N'] == x_val, 'duration'].values[0]) / 2
        x_values.append(x_val)
        y_values.append(curr_y_value)
        # counter += 1

    return x_values, y_values


def get_matrix_size_accesses_time_per_access(x_val, y_val):
    init = pd.DataFrame(zip(x_val, y_val), columns=['N', 'duration'])
    init.sort_values(by=['N'], inplace=True)
    init.reset_index(drop=True, inplace=True)
    init['matrix size'] = init['N'] ** 2
    init['accesses'] = init['N'] ** 3
    init['time per access'] = init['duration'] / init['accesses']
    return init


def scatter_plotting(x_val, y_val, filename, x_axis_name, y_axis_name, curve_func=None, curve_names=None, colors=None, args=None):
    if curve_names is None:
        curve_names = []
    if colors is None:
        colors = []
    if curve_func is None:
        curve_func = []
    if args is None:
        args = [[]]
    print(f'all args: {args}')
    plt.figure(figsize=(10, 6))
    plt.scatter(x_val, y_val, color='blue')
    plt.ylim(ymin=0)
    for i, curve in enumerate(curve_names):
        print(f"args passed in {curve}: {args[i]}")
        wh_plot = True
        if curve in ['sqrt', 'log']:
            wh_plot = False
        curve_plotting(x_val, curve_func[i], curve, colors[i], args[i], wh_plot=wh_plot)
        print(f"finished {curve}")

    plt.title(f'{x_axis_name} vs {y_axis_name}')
    plt.xlabel(f'{x_axis_name}')
    plt.ylabel(f'{y_axis_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=100)
    plt.show()


def curve_plotting(x_val, func, curve_name, color, *args, wh_plot=True):
    [a] = args
    # print(f"args: {args}")
    tail = "".join([f"{i}={word:.2e} " for i, word in zip(string.ascii_lowercase, *args)])
    # print(f"{curve_name}: {tail}")
    # print(f"a: {a}")

    if wh_plot is True:
        w = []
        h = []

        for i in np.arange(min(x_val), max(x_val), 1):
            w.append(i)
            h.append(func(i, *a))
        plt.plot(w, h, label=f'{curve_name}: ' + tail, color=color)
    else:
        plt.plot(x_val, func(x_val, *a), label=f'{curve_name}: ' + tail, color=color)


def print_curve_fit_result(popt, name_of_curve, pcov=None):
    for i, word in zip(string.ascii_lowercase, popt):
        print(f"{name_of_curve} {i}: {word:.2e}")
        if pcov is not None:
            print(f"covariance: {pcov[i]}")
    print()


def get_r2_score(curve_name, x_val, y_val, func, *args):
    print(f"R^2 value for {curve_name}: {r2_score([i for i in x_val], [func(i, *args) for i in y_val])}")


def fit_curve(x_val, y_val, func):
    # pack x and y into a dataframe
    total = pd.DataFrame(zip(x_val, y_val), columns=['X', 'Y'])
    total.sort_values(by=['X'], inplace=True)
    total.reset_index(drop=True, inplace=True)
    # type check
    if total['X'].dtype != np.float64:
        # print(f"type of total['X']: {total['X'].dtype}")
        total['X'] = total['X'].astype(np.float64)
    if total['Y'].dtype != np.float64:
        # print(f"type of total['Y']: {total['Y'].dtype}")
        total['Y'] = total['Y'].astype(np.float64)
    # print('here')
    popt, pcov = opt.curve_fit(func, total['X'], total['Y'], check_finite=True, nan_policy='raise')
    return popt, pcov


def parse_cache_bandwidth(file_name):
    pattern1 = (r"Array Size: (\d+) elements in double\n")
    pattern2 = (r"(CYCLIC|RAND CYCLIC|SAWTOOTH|RAND SAWTOOTH): Best Rate MB\/s (\d+\.?\d*e?[+-]?\d*|inf), Avg time (\d+\.?\d*e?[+-]?\d*), Min time (\d+\.?\d*e?[+-]?\d*), Max time (\d+\.?\d*e?[+-]?\d*), Access Times(\d+\.?\d*e?[+-]?\d*), Avg Time per Access(\d+\.?\d*e?[+-]?\d*)")
    with open(file_name, "r") as file:
        text = file.read()
        matches = re.findall([pattern1, pattern2], text)
        print(matches)

        rtn_df = {
            "CYCLIC": pd.DataFrame(
            {
                "Per Array Size": [],
                "Avg time": [], 
                "Access Times": [], 
                "Avg Time per Access": []
            }
        ), 
            "RAND CYCLIC": pd.DataFrame(
            {
                "Per Array Size": [],
                "Avg time": [], 
                "Access Times": [], 
                "Avg Time per Access": []
            }
        ),
            "SAWTOOTH": pd.DataFrame(
            {
                "Per Array Size": [],
                "Avg time": [],
                "Access Times": [],
                "Avg Time per Access": []
            }
        ),
            "RAND SAWTOOTH": pd.DataFrame(
            {
                "Per Array Size": [],
                "Avg time": [],
                "Access Times": [],
                "Avg Time per Access": []
            }
        )
        }
        n_times = 10
        pprint(matches)    

        for match in matches:
            array_size = int(match[0])
            acc_pat = match[1]
            avg_time = float(match[3])
            min_time = float(match[4])
            max_time = float(match[5])
            access_times = math.ceil(array_size/8) * 3
            avg_time_per_access = avg_time / access_times
            cache_df = rtn_df.get(acc_pat)
            # cache_df.concat([
            #     {"Per Array Size": array_size}, 
            #     {"Avg time": avg_time},
            #     {"Access Times": access_times},
            #     {"Avg Time per Access": avg_time_per_access}
            #     ])
            cache_df.loc[len(cache_df)] = {
                "Per Array Size": array_size,
                "Avg time": avg_time, 
                "Access Times": access_times, 
                "Avg Time per Access": avg_time_per_access
            }
        #     tmp_data = [array_size, avg_time, access_times, avg_time_per_access]
        #     print(tmp_data)
        #     rtn_df.update({language: pd.DataFrame(tmp_data, columns=["Per Array Size", "Avg time", "Access Times", "Avg Time per Access"])})
        # print(rtn_df)
        # Create a DataFrame with the extracted data
        # cache_bandwidth_df = pd.DataFrame(cache_bandwidth_data, columns=["Per Array Size", "Avg time", "Access Times", "Avg Time per Access"])
        return rtn_df


def main():
    def polynomial(x, u, b, c, d, e):
        return u * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

    def quadratic(x, a, b, c):
        return a * x ** 2 + b * x + c

    def cubic_with_term(x, a, b, c, d):
        return a * x ** 3 + b * x ** 2 + c * x + d

    def cubic(x, a):
        return a * x ** 3

    def fourth(x, a):
        return a * x ** 4

    # fit a sqrt curve in the plot
    def sqrt_fit(x, a):
        return a * np.sqrt(x)

    def log_fit(x, a, b):
        return a * np.log(b * x)

    def sigmoid(x, L, x0, k, b):
        # if -x > np.log(np.finfo(type(x)).max):
        #     return 0.0
        # if the result of the sigmoid function is too large for overflow, return the max value of float64
        # rtn = L / (1 + np.exp(-k * (x - x0))) + b
        rtn = L/(1 + np.exp(-(k*x)))+b
        # print(f'x: {x}, L: {L}, x0: {x0}, k: {k}, b: {b}')
        # print(f"rtn: {rtn}")
        if min(rtn) < 1.7e-308:
            # rtn = np.float64.min
            print(f"too small")
            print(f"min: {min(rtn)}")
            print(f'x: {x}, L: {L}, x0: {x0}, k: {k}, b: {b}')
            print(f'rtn: {rtn}')
            # index = rtn[rtn < 1.7e-308].index
            # index = np.where(rtn < 1.7e-308)
            # index = rtn.index(min(rtn))
            # print(f"rank: {index}")
        if max(rtn) > 1.7e+308:
            # rtn = np.float64.max
            print(f"too large")
        return rtn

    def simplier_sigmoid(x, a, b, c):
        return 1/(1 + np.exp(-x))

    def tanh(x, a):
        # TODO: need to finish tanh
        pass

    def ln(x, a, b):
        return a * np.log(b * x)

    def rand_evict(c, d):
        return 1 - (1 - (1 / c)) ** d

    def rand_evict_sawtooth(c, d):
        sum = 0
        for i in range(1, d):
            sum += (1 - (1 - (1 / c)) ** i)
    rd_cr = [0, 0.445, 0.911, 0.995, 0.999]
    rd_cr = [0, 0.445*340, 0.911*340, 0.995*340, 0.999*340]
    rd_d = [64*Mi, 128*Mi, 256*Mi, 512*Mi, 1*Gi]
    # mr_with_set_associativity = [0, 0.415, 0.903, ]

    def new_rand_evict(x, c, d):
        return 1 - (1 - 1 / c) ** (d * x)

    SECOND_CURVE = False
    RAND_EVICTION_CURVE = False
    CONVERT_TO_SEC = False
    SCALE_SECOND_CURVE = False # scale the second curve to the first curve
    TRIM_DATA = True
    CURVE_FITTING = False
    DATA_PLOTTING = True
    CACHE_SIZE_PLOTTING = True
    YIFAN_CACHE_SZIE_PLOTTING = False
    SHAOTONG_CACHE_SIZE_PLOTTING = True
    CYCLE2_CACHE_SIZE_PLOTTING = False
    PRINT_DATA = True
    DIFFERENCE_IN_RESULTS = False

    yifan_l2_cache = 0.5*Mi # private L2 cache
    yifan_l3_cache = 96*Mi
    shaotong_l2_cache = 1*Mi # private L2 cache
    shaotong_l3_cache = 96*Mi
    shaotong_l3_cache_no3d = 32*Mi
    cycle2_l2_cache = 0.25*Mi # 3Mib(12 instances) L2 cache
    cycle2_l3_cache = 15*Mi # 30Mib(2 instances) L3 cache
    AWS_l2_cache = 8*Mi
    AWS_l3_cache = 32*Mi

    yifan_peak_freq = 3.5 # GHz
    shaotong_peak_freq = 5.7 # GHz

    plt.figure(figsize=(10, 6))

    cache_log_file_name = "shaotong_default_dependent_large_scale.csv"
    cache_df = pd.read_csv(cache_log_file_name)

    dfs = {}
    for access_pattern in cache_df['Access Pattern'].unique():
        dfs[access_pattern] = cache_df[cache_df['Access Pattern'] == access_pattern].copy()

    if SECOND_CURVE:
        second_cache_log_file_name = "yifan_huge_pg_2MiB_dependent.csv"
        second_cache_df = pd.read_csv(second_cache_log_file_name)

        second_dfs = {}
        for access_pattern in cache_df['Access Pattern'].unique():
            second_dfs[access_pattern] = second_cache_df[second_cache_df['Access Pattern'] == access_pattern].copy()

    cyclic_df = dfs['CYCLIC'].groupby('Size')['Avg Time per Access'].mean().reset_index()
    sawtooth_df = dfs['SAWTOOTH'].groupby('Size')['Avg Time per Access'].mean().reset_index()
    for_for_df = dfs['RAND FORWARD FORWARD'].groupby('Size')['Avg Time per Access'].mean().reset_index()
    for_back_df = dfs['RAND FORWARD BACKWARD'].groupby('Size')['Avg Time per Access'].mean().reset_index()
    back_back_df = dfs['RAND BACKWARD BACKWARD'].groupby('Size')['Avg Time per Access'].mean().reset_index()

    if SECOND_CURVE:
        second_cyclic_df = second_dfs['CYCLIC'].groupby('Size')['Avg Time per Access'].mean().reset_index()
        second_sawtooth_df = second_dfs['SAWTOOTH'].groupby('Size')['Avg Time per Access'].mean().reset_index()
        second_for_for_df = second_dfs['RAND FORWARD FORWARD'].groupby('Size')['Avg Time per Access'].mean().reset_index()
        second_for_back_df = second_dfs['RAND FORWARD BACKWARD'].groupby('Size')['Avg Time per Access'].mean().reset_index()
        second_back_back_df = second_dfs['RAND BACKWARD BACKWARD'].groupby('Size')['Avg Time per Access'].mean().reset_index()

        if SCALE_SECOND_CURVE:
            cyclic_scale = cyclic_df['Avg Time per Access'].max() / second_cyclic_df['Avg Time per Access'].max()
            sawtooth_scale = sawtooth_df['Avg Time per Access'].max() / second_sawtooth_df['Avg Time per Access'].max()
            for_for_scale = for_for_df['Avg Time per Access'].max() / second_for_for_df['Avg Time per Access'].max()
            for_back_scale = for_back_df['Avg Time per Access'].max() / second_for_back_df['Avg Time per Access'].max()
            back_back_scale = back_back_df['Avg Time per Access'].max() / second_back_back_df['Avg Time per Access'].max()

    if TRIM_DATA:
        cyclic_df = cyclic_df.loc[cyclic_df['Size'] <= 1*Gi / 8]
        sawtooth_df = sawtooth_df.loc[sawtooth_df['Size'] <= 1*Gi / 8]
        for_for_df = for_for_df.loc[for_for_df['Size'] <= 1*Gi  / 8]
        for_back_df = for_back_df.loc[for_back_df['Size'] <= 1*Gi / 8]
        back_back_df = back_back_df.loc[back_back_df['Size'] <= 1*Gi / 8]
        if SECOND_CURVE:
            second_cyclic_df = second_cyclic_df.loc[second_cyclic_df['Size'] <= 128*Mi / 8]
            second_sawtooth_df = second_sawtooth_df.loc[second_sawtooth_df['Size'] <= 128*Mi / 8]
            second_for_for_df = second_for_for_df.loc[second_for_for_df['Size'] <= 128*Mi / 8]
            second_for_back_df = second_for_back_df.loc[second_for_back_df['Size'] <= 128*Mi / 8]
            second_back_back_df = second_back_back_df.loc[second_back_back_df['Size'] <= 128*Mi/ 8]

    if DIFFERENCE_IN_RESULTS:
        difference_in_cyclic = cyclic_df['Avg Time per Access'] - second_cyclic_df['Avg Time per Access']
        difference_in_sawtooth = sawtooth_df['Avg Time per Access'] - second_sawtooth_df['Avg Time per Access']
        difference_in_for_for = for_for_df['Avg Time per Access'] - second_for_for_df['Avg Time per Access']
        difference_in_for_back = for_back_df['Avg Time per Access'] - second_for_back_df['Avg Time per Access']
        difference_in_back_back = back_back_df['Avg Time per Access'] - second_back_back_df['Avg Time per Access']

        difference_ratio_in_cyclic = difference_in_cyclic / cyclic_df['Avg Time per Access']
        difference_ratio_in_sawtooth = difference_in_sawtooth / sawtooth_df['Avg Time per Access']
        difference_ratio_in_for_for = difference_in_for_for / for_for_df['Avg Time per Access']
        difference_ratio_in_for_back = difference_in_for_back / for_back_df['Avg Time per Access']
        difference_ratio_in_back_back = difference_in_back_back / back_back_df['Avg Time per Access']

    if RAND_EVICTION_CURVE:
        # rand_evict_scale = statistics.mean([for_for_df['Avg Time per Access'].max(), for_back_df['Avg Time per Access'].max(),
        #                                     back_back_df['Avg Time per Access'].max()]) / rand_evict(96*Mi / 64, (cyclic_df['Size'].max() - ((32*Ki + 1*Mi) / 8)) / 8)
        rand_evict_scale = statistics.mean([for_for_df['Avg Time per Access'].max(), for_back_df['Avg Time per Access'].max(),
                                            back_back_df['Avg Time per Access'].max()]) / 1

        if SECOND_CURVE:
            second_evict_scale = statistics.mean([second_for_for_df['Avg Time per Access'].max(), second_for_back_df['Avg Time per Access'].max(),
                                                    second_back_back_df['Avg Time per Access'].max()]) / rand_evict(32 * Ki + 0.5 * Mi + 96 * Mi, second_cyclic_df['Size'].max() * 8)

    if CONVERT_TO_SEC:
        cyclic_df['Avg Time per Access'] = cyclic_df['Avg Time per Access'] / shaotong_peak_freq
        sawtooth_df['Avg Time per Access'] = sawtooth_df['Avg Time per Access'] / shaotong_peak_freq
        for_for_df['Avg Time per Access'] = for_for_df['Avg Time per Access'] / shaotong_peak_freq
        for_back_df['Avg Time per Access'] = for_back_df['Avg Time per Access'] / shaotong_peak_freq
        back_back_df['Avg Time per Access'] = back_back_df['Avg Time per Access'] / shaotong_peak_freq

        if SECOND_CURVE:
            second_cyclic_df['Avg Time per Access'] = second_cyclic_df['Avg Time per Access'] / yifan_peak_freq
            second_sawtooth_df['Avg Time per Access'] = second_sawtooth_df['Avg Time per Access'] / yifan_peak_freq
            second_for_for_df['Avg Time per Access'] = second_for_for_df['Avg Time per Access'] / yifan_peak_freq
            second_for_back_df['Avg Time per Access'] = second_for_back_df['Avg Time per Access'] / yifan_peak_freq
            second_back_back_df['Avg Time per Access'] = second_back_back_df['Avg Time per Access'] / yifan_peak_freq

    if PRINT_DATA:
        if DIFFERENCE_IN_RESULTS:
            print(f"difference_in_cyclic: \n{difference_in_cyclic}")
            print(f"difference_in_sawtooth: \n{difference_in_sawtooth}")
            print(f"difference_in_for_for: \n{difference_in_for_for}")
            print(f"difference_in_for_back: \n{difference_in_for_back}")
            print(f"difference_in_back_back: \n{difference_in_back_back}")

            print(f"difference_ratio_in_cyclic: \n{difference_ratio_in_cyclic}")
            print(f"difference_ratio_in_sawtooth: \n{difference_ratio_in_sawtooth}")
            print(f"difference_ratio_in_for_for: \n{difference_ratio_in_for_for}")
            print(f"difference_ratio_in_for_back: \n{difference_ratio_in_for_back}")
            print(f"difference_ratio_in_back_back: \n{difference_ratio_in_back_back}")
        else:
            print(f"back back: {back_back_df}")
            print(f"for for: {for_for_df}")
            print(f"for back: {for_back_df}")
            print(f"cyclic: {cyclic_df}")
            print(f"sawtooth: {sawtooth_df}")

            if SECOND_CURVE:
                print(f"second back back: {second_back_back_df}")
                print(f"second for for: {second_for_for_df}")
                print(f"second for back: {second_for_back_df}")
                print(f"second cyclic: {second_cyclic_df}")
                print(f"second sawtooth: {second_sawtooth_df}")

    if RAND_EVICTION_CURVE:
        print(f"rand_evict_scale: {rand_evict_scale}")
        if SECOND_CURVE:
            print(f"second_evict_scale: {second_evict_scale}")

    if CURVE_FITTING:
        cyclic_popt, cyclic_pcov = fit_curve(cyclic_df['Size']*8, cyclic_df['Avg Time per Access'], sqrt_fit)
        sawtooth_popt, sawtooth_pcov = fit_curve(sawtooth_df['Size']*8, sawtooth_df['Avg Time per Access'], sqrt_fit)
        for_for_popt, for_for_pcov = fit_curve(for_for_df['Size']*8, for_for_df['Avg Time per Access'], sqrt_fit)
        for_back_popt, for_back_pcov = fit_curve(for_back_df['Size']*8, for_back_df['Avg Time per Access'], sqrt_fit)
        back_back_popt, back_back_pcov = fit_curve(back_back_df['Size']*8, back_back_df['Avg Time per Access'], sqrt_fit)

        print_curve_fit_result(cyclic_popt, 'cyclic')
        print_curve_fit_result(sawtooth_popt, 'sawtooth')
        print_curve_fit_result(for_for_popt, 'for_for')
        print_curve_fit_result(for_back_popt, 'for_back')
        print_curve_fit_result(back_back_popt, 'back_back')

        plt.plot(cyclic_df['Size']*8, sqrt_fit(cyclic_df['Size']*8, *cyclic_popt), label='fitted cyclic', color='red', linestyle='dashed')
        plt.plot(sawtooth_df['Size']*8, sqrt_fit(sawtooth_df['Size']*8, *sawtooth_popt), label='fitted sawtooth', color='green', linestyle='dashed')
        plt.plot(for_for_df['Size']*8, sqrt_fit(for_for_df['Size']*8, *for_for_popt), label='fitted for_for', color='blue', linestyle='dashed')
        plt.plot(for_back_df['Size']*8, sqrt_fit(for_back_df['Size']*8, *for_back_popt), label='fitted for_back', color='yellow', linestyle='dashed')
        plt.plot(back_back_df['Size']*8, sqrt_fit(back_back_df['Size']*8, *back_back_popt), label='fitted back_back', color='black', linestyle='dashed')
        # may need to fit the second curve as well

    if RAND_EVICTION_CURVE:
        plt.plot(cyclic_df['Size']*8, (rand_evict(32*Ki + 1*Mi + 128*Mi, cyclic_df['Size']*8)) * rand_evict_scale, label='7950X3D random eviction', color='black', linestyle='dashed', linewidth=5)
        if SECOND_CURVE:
            plt.plot(second_cyclic_df['Size']*8, (rand_evict(32*Ki + 0.5*Mi + 768*Mi, second_cyclic_df['Size']*8)) * second_evict_scale, label='7773X random eviction', color='red', linestyle='dashed', linewidth=5)

    if DATA_PLOTTING:
        if DIFFERENCE_IN_RESULTS:
            plt.plot(cyclic_df['Size']*8, difference_in_cyclic, label='diff in cyclic', color='red', linewidth=3)
            plt.plot(sawtooth_df['Size']*8, difference_in_sawtooth, label='diff in sawtooth', color='green', linewidth=3)
            plt.plot(for_for_df['Size']*8, difference_in_for_for, label='diff in for_for', color='blue', linewidth=3)
            plt.plot(for_back_df['Size']*8, difference_in_for_back, label='diff in for_back', color='yellow', linewidth=3)
            plt.plot(back_back_df['Size']*8, difference_in_back_back, label='diff in back_back', color='black', linewidth=3)

            plt.plot(cyclic_df['Size']*8, difference_ratio_in_cyclic, label='diff ratio in cyclic', color='red', linewidth=3, linestyle='dashed')
            plt.plot(sawtooth_df['Size']*8, difference_ratio_in_sawtooth, label='diff ratio in sawtooth', color='green', linewidth=3, linestyle='dashed')
            plt.plot(for_for_df['Size']*8, difference_ratio_in_for_for, label='diff ratio in for_for', color='blue', linewidth=3, linestyle='dashed')
            plt.plot(for_back_df['Size']*8, difference_ratio_in_for_back, label='diff ratio in for_back', color='yellow', linewidth=3, linestyle='dashed')
            plt.plot(back_back_df['Size']*8, difference_ratio_in_back_back, label='diff ratio in back_back', color='black', linewidth=3, linestyle='dashed')
        else:
            # plt.plot(cyclic_df['Size']*8, cyclic_df['Avg Time per Access'], label='cyclic', color='red', linewidth=3)
            # plt.plot(sawtooth_df['Size']*8, sawtooth_df['Avg Time per Access'], label='sawtooth', color='green', linewidth=3)
            plt.plot(for_for_df['Size']*8, for_for_df['Avg Time per Access'], label='7950X3D for_for', color='blue', linewidth=3)
            plt.plot(for_back_df['Size']*8, for_back_df['Avg Time per Access'], label='7950X3D for_back', color='yellow', linewidth=3)
            plt.plot(back_back_df['Size']*8, back_back_df['Avg Time per Access'], label='7950X3D back_back', color='black', linewidth=3)

            if SECOND_CURVE:
                # plt.plot(second_cyclic_df['Size']*8, second_cyclic_df['Avg Time per Access'], label='1GiB Huge Page cyclic', color='red', linewidth=3, linestyle='dashed')
                # plt.plot(second_sawtooth_df['Size']*8, second_sawtooth_df['Avg Time per Access'], label='1GiB Huge Page sawtooth', color='green', linewidth=3, linestyle='dashed')
                plt.plot(second_for_for_df['Size']*8, second_for_for_df['Avg Time per Access'], label='7773X for_for', color='blue', linewidth=3, linestyle='dashed')
                plt.plot(second_for_back_df['Size']*8, second_for_back_df['Avg Time per Access'], label='7773X for_back', color='yellow', linewidth=3, linestyle='dashed')
                plt.plot(second_back_back_df['Size']*8, second_back_back_df['Avg Time per Access'], label='7773X back_back', color='black', linewidth=3, linestyle='dashed')


    if CACHE_SIZE_PLOTTING:
        plt.axvline(x=32*Ki, color='black', label='L1 size')
        if SHAOTONG_CACHE_SIZE_PLOTTING:
            plt.axvline(x=shaotong_l2_cache, color='black', label='7950X3D L2 size')
            plt.axvline(x=shaotong_l3_cache, color='black', label='L3 size(7950X3D CCD0&7773X)')

        if YIFAN_CACHE_SZIE_PLOTTING:
            plt.axvline(x=yifan_l2_cache, color='red', label='7773X L2 size')
            # plt.axvline(x=yifan_l3_cache, color='red', label='7773X L3 size')

        if CYCLE2_CACHE_SIZE_PLOTTING:
            plt.axvline(x=cycle2_l2_cache, color='green', label='E5-2430 L2 size')
            plt.axvline(x=cycle2_l3_cache, color='green', label='E5-2430 L3 size')

    ax = plt.gca()
    # ax.set_ylim([0, 20])
    # ax.set_xlim([0, 0.5*Mi])

    # plt.xscale('log', base=2)
    # plt.rcParams.update({'font.size': 13})
    # plt.rc('figure', titlesize=50)
    # plt.rc('legend', fontsize=22)
    # plt.rc('axes', titlesize=30)
    # plt.rc('axes', labelsize=30)
    # plt.rc('xtick', labelsize=30)
    # plt.rc('ytick', labelsize=30)

    # ax.tick_params(axis='x', labelsize=18)
    # ax.tick_params(axis='y', labelsize=18)

    if CONVERT_TO_SEC:
        plt.title('Total Arrays Size vs. Avg latency per Access(ns)', fontsize=27)
        plt.ylabel('Avg latency per Access(ns)', fontsize=22)
    else:
        plt.title('Total Arrays Size vs. Avg latency per Access(cycle)', fontsize=27)
        plt.ylabel('Avg latency per Access(cycle)', fontsize=22)

    plt.plot(rd_d, rd_cr , label='7950X3D Random Eviction', color='black', linestyle='dashed', linewidth=5)

    plt.xlabel('Total Arrays Size(B)', fontsize=22)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.savefig(cache_log_file_name + '_time_per_access.png', dpi=100)
    plt.show()
    print(rand_evict(96*Mi / 64, (128*Mi - 32*Ki - Mi) / 64))


if __name__ == '__main__':
    main()
