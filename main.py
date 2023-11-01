import math
import re
import string

import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.metrics import r2_score

# TODO: do the 1-2400 corun on server
# TODO: write a script to test where does the O1 to O3 optimization difference decrease to 0
opt_gap_filename = 'sawtooth_opt-gap-log'
co_run_log_filename = 'aarch64-cloud-machine_1-3000sawtooth_O3gemm_3hr_log'


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
            args_match = re.search(r'args:\s*Some\(\s*\[\s*([\s\S]*?)\s*\],\s*\)', match)
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

        for i in np.arange(min(x_val), max(x_val), 0.01):
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
    pattern = r"Array Size: (\d+) elements\(\dB double\) in (rust|c)\nTriad: Best Rate MB\/s (\d+\.?\d*e?[+-]?\d*), Avg time (\d+\.?\d*e?[+-]?\d*), Min time (\d+\.?\d*e?[+-]?\d*), Max time (\d+\.?\d*e?[+-]?\d*), Access Times(\d+\.?\d*e?[+-]?\d*), Avg Time per Access(\d+\.?\d*e?[+-]?\d*)"
    with open(file_name, "r") as file:
        text = file.read()
        matches = re.findall(pattern, text)

        cache_bandwidth_data = []
        rtn_df = {"rust": pd.DataFrame(
            {
                "Per Array Size": [],
                "Avg time": [], 
                "Access Times": [], 
                "Avg Time per Access": []
            }
        ), 
        "c": pd.DataFrame(
            {
                "Per Array Size": [],
                "Avg time": [], 
                "Access Times": [], 
                "Avg Time per Access": []
            }
        )}
        n_times = 10
        pprint(matches)    

        for match in matches:
            array_size = int(match[0])
            language = match[1]
            avg_time = float(match[3])
            access_times = array_size * 3
            avg_time_per_access = avg_time / access_times
            cache_df = rtn_df.get(language)
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
    # read in the txt file
    # extracted_data = parsing_log()
    # non_avg_x, non_avg_y = parse_extraced_data(extracted_data)
    # zip_to_csv(non_avg_x, non_avg_y, co_run_log_filename+"_non-avg")
    #
    # # get the min and max value in data['N']
    # min_n = min([data['N'] for data in extracted_data])
    # max_n = max([data['N'] for data in extracted_data])
    #
    # # get the avg of duration for each N
    # x_values, y_values = averaging_size_and_duration(extracted_data)
    # time_per_access_df = get_matrix_size_accesses_time_per_access(non_avg_x, non_avg_y)
    # time_per_access_df.to_csv(co_run_log_filename + '_time-per-access.csv', index=False)
    # # print(time_per_access_df)

    # fit the curve
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

    # avg_copt, _ = fit_curve(x_values, y_values, cubic)
    # avg_cwtopt, _ = fit_curve(x_values, y_values, cubic_with_term)
    # avg_fopt, _ = fit_curve(x_values, y_values, fourth)
    # non_avg_copt, _ = fit_curve(non_avg_x, non_avg_y, cubic)
    # non_avg_cwtopt, _ = fit_curve(non_avg_x, non_avg_y, cubic_with_term)
    # non_avg_fopt, _ = fit_curve(non_avg_x, non_avg_y, fourth)
    # tpa_sqrt_opt, _ = fit_curve(time_per_access_df['matrix size'], time_per_access_df['time per access'], sqrt_fit)
    # tpa_log_opt, _ = fit_curve(time_per_access_df['matrix size'], time_per_access_df['time per access'], log_fit)
    # tpa_sigmoid_opt, _ = fit_curve(time_per_access_df['matrix size'], time_per_access_df['time per access'], sigmoid)
    #
    # print_curve_fit_result(avg_copt, 'averaged cubic')
    # print_curve_fit_result(avg_cwtopt, 'averaged cubic with term')
    # print_curve_fit_result(avg_fopt, 'averaged fourth')
    # print_curve_fit_result(non_avg_copt, 'non-averaged cubic')
    # print_curve_fit_result(non_avg_cwtopt, 'non-averaged cubic with term')
    # print_curve_fit_result(non_avg_fopt, 'non-averaged fourth')
    # print_curve_fit_result(tpa_sqrt_opt, 'time per access sqrt')
    # print_curve_fit_result(tpa_log_opt, 'time per access log')
    # print_curve_fit_result(tpa_sigmoid_opt, 'time per access sigmoid')
    #
    # # draw the plot
    # scatter_plotting(time_per_access_df['matrix size'], time_per_access_df['time per access'], co_run_log_filename + '_time-per-access',
    #                  'Matrix Size', 'Time per Access (s)',
    #                  [sqrt_fit, log_fit], ['sqrt', 'log'], ['green', 'red'], [[*tpa_sqrt_opt], [*tpa_log_opt]])
    #
    # scatter_plotting(x_values, y_values, co_run_log_filename + '_avged',
    #                  'N', 'Duration (s)',
    #                  [cubic, cubic_with_term, fourth], ['cubic', 'cubic with terms', 'fourth'], ['green', 'red', 'yellow'], [[*avg_copt], [*avg_cwtopt], [*avg_fopt]])
    #
    # scatter_plotting(non_avg_x, non_avg_y, co_run_log_filename + '_non-avg',
    #                  'N', 'Duration (s)',
    #                  [cubic, cubic_with_term, fourth], ['cubic', 'cubic with terms', 'fourth'], ['green', 'red', 'yellow'], [[*non_avg_copt], [*non_avg_cwtopt], [*non_avg_fopt]])

    cache_log_file_name = "cache_bandwidth_log"
    plt.figure(figsize=(10, 6))
    cache_df = parse_cache_bandwidth(cache_log_file_name)
    print(cache_df)
    c_df = cache_df["c"]
    rust_df = cache_df["rust"]
    plt.scatter(c_df['Per Array Size'], c_df['Avg Time per Access'], color='blue')
    plt.scatter(rust_df['Per Array Size'], rust_df['Avg Time per Access'], color='red')
    plt.title('Triad per Array Size vs. Avg Time per Access(s)')
    plt.xlabel('Triad per Array Size')
    plt.ylabel('Avg Time per Access(s)')
    plt.grid(True)
    plt.savefig(cache_log_file_name + '_time_per_access', dpi=100)
    plt.show()


    # plt.figure(figsize=(10, 6))
    # time_per_access_df = time_per_access_df[time_per_access_df['time per access'] != 1e-6]
    # print(time_per_access_df)
    # plt.scatter(time_per_access_df['matrix size'], time_per_access_df['time per access'], color='blue')
    # plt.title('Matrix Size vs Time per Access (s)')
    # plt.xlabel('Matrix Size')
    # plt.ylabel('Time per Access (s)')
    # plt.grid(True)
    # plt.savefig(co_run_log_filename + '_time_per_access', dpi=100)
    # plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # plt.scatter(non_avg_x, non_avg_y, color='blue')
    # plt.title('Matrix Size vs Time per Access (s)')
    # plt.xlabel('Matrix Size')
    # plt.ylabel('Time per Access (s)')
    # plt.grid(True)
    # plt.savefig(co_run_log_filename + '_non_avged', dpi=100)
    # plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # plt.scatter(x_values, y_values, color='blue')
    # plt.title('Matrix Size vs Time per Access (s)')
    # plt.xlabel('Matrix Size')
    # plt.ylabel('Time per Access (s)')
    # plt.grid(True)
    # plt.savefig(co_run_log_filename + '_avged', dpi=100)
    # plt.show()

    # w = []
    # h = []
    #
    # for i in np.arange(min_n, max_n, 0.01):
    #     w.append(i)
    #     h.append(polynormial(i, popt[0], popt[1], popt[2], popt[3], popt[4]))
    #
    # qw = []
    # qh = []
    # for i in np.arange(min_n, max_n, 0.01):
    #     qw.append(i)
    #     qh.append(quadratic(i, qopt[0], qopt[1], qopt[2]))
    #
    # cwtw = []
    # cwth = []
    # for i in np.arange(min_n, max_n, 0.01):
    #     cwtw.append(i)
    #     cwth.append(cubic_with_term(i, cwtopt[0], cwtopt[1], cwtopt[2], cwtopt[3]))
    #
    # cw = []
    # ch = []
    #
    # for i in np.arange(min_n, max_n, 0.01):
    #     cw.append(i)
    #     ch.append(cubic(i, copt[0]))
    #
    # fw = []
    # fh = []
    #
    # for i in np.arange(min_n, max_n, 0.01):
    #     fw.append(i)
    #     fh.append(fourth(i, fopt[0]))




        # Create the plot
        # plt.figure(figsize=(10, 6))
        # # plt.ion()
        # plt.scatter(x_values, y_values, color='blue')
        # # plt.ioff()
        # plt.title('N vs Duration (s)')
        # plt.xlabel('N')
        # plt.ylabel('Duration (s)')
        # # plt.plot(w, h, color='red')
        # # plt.plot(qw, qh, color='purple')
        # plt.plot(cwtw, cwth, color='orange')
        # plt.plot(cw, ch, color='yellow')
        # plt.plot(fw, fh, color='green')
        # plt.grid(True)
        # plt.savefig(co_run_log_filename, dpi=100)
        # plt.show()
        #
        # # Draw another plot using the abnormal points
        # # plt.figure(figsize=(10, 6))
        # # plt.scatter([i[0] for i in abnormal_points], [i[1] for i in abnormal_points], color='blue')
        # # plt.title('N vs Duration (s) with abnormal points')
        # # plt.xlabel('N')
        # # plt.ylabel('Duration (s)')
        # # plt.plot(abn_fw, abn_fh, color='green')
        # # plt.plot(abn_cwtw, abn_cwth, color='orange')
        # # plt.grid(True)
        # # plt.savefig(co_run_log_filename + '_abnormal', dpi=100)
        # plt.show()

        # # Draw another plot without the average
        # plt.figure(figsize=(10, 6))
        # plt.scatter(non_avg_x, non_avg_y, color='blue')
        # plt.title('N vs Duration (s) without average')
        # plt.xlabel('N')
        # plt.ylabel('Duration (s)')
        # plt.grid(True)
        # plt.savefig(co_run_log_filename + '_noavg', dpi=100)
        # plt.show()



if __name__ == '__main__':
    main()
