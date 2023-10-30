import math
import re
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
import pandas as pd
from pprint import pprint

from numpy._typing._extended_precision import float256
from sklearn.metrics import r2_score

time_per_access_filename = 'correct_1-5000sawtooth_O3gemm_5hr_with_time-per-access'

# TODO: 1. fit a curve(maybe sigmoid) to the data points up to n=2200
#       (with x-axis as data size and y-axis as time per access)
# TODO: 2. share result with wes and get his number and his plot (√)
# TODO: 3. run sawtooth on other cloud machines(√) and fit sqrt curve to the data points

# TODO: for sometimes later...: try running the random tile-sized recursive matrix multiplication algorithm with and without sawtooth


def main():
    time_per_access = pd.read_csv(time_per_access_filename+'.csv')
    # print(time_per_access)
    time_per_access['data size'] = time_per_access['data size'].astype(float)
    # get rid of the data size that's larger than 7.0E+06 and the corresponding time per access
    time_per_access = time_per_access[time_per_access['data size'] <= 4.0e+06]

    # fit a sqrt curve in the plot
    def sqrt_fit(x, a):
        return a * np.sqrt(x)

    def log_fit(x, a, b):
        return a * np.log(b * x)

    # def sigmoid(x, L, x0, k, b):
    #     if -x > np.log(np.finfo(type(x)).max):
    #         return 0.0
    #     return L / (1 + np.exp(-k * (x - x0))) + b
    #
    # p0 = [max(time_per_access['time per access']),
    #       np.median(time_per_access['data size']),
    #       1,
    #       min(time_per_access['time per access'])]  # this is an mandatory initial guess

    # sigmoid_popt, sigmoid_pcov = opt.curve_fit(sigmoid, time_per_access['data size'],
    #                                            time_per_access['time per access'], p0, method='dogbox')

    sqrt_popt, _ = opt.curve_fit(sqrt_fit, time_per_access['data size'], time_per_access['time per access'])

    log_popt, _ = opt.curve_fit(log_fit, time_per_access['data size'], time_per_access['time per access'])
    print(f'log a: {log_popt}')

    # sigmoid_popt, _ = opt.curve_fit(sigmoid, time_per_access['data size'], time_per_access['time per access'])
    # print(f'sigmoid a: {sigmoid_popt}')



    plt.figure(figsize=(10, 6))
    plt.scatter(time_per_access['data size'], time_per_access['time per access'], marker='o', label='Data Points', color='blue')
    plt.plot(time_per_access['data size'], sqrt_fit(time_per_access['data size'], sqrt_popt[0]), label=f'sqrt: a = {sqrt_popt[0]:.2e}', color='green')
    plt.plot(time_per_access['data size'], log_fit(time_per_access['data size'], log_popt[0], log_popt[1]), label=f'log with term: a = {log_popt[0]:.2e}, b = {log_popt[1]:.2e}', color='red')
    # plt.plot(time_per_access['data size'], sigmoid(time_per_access['data size'], sigmoid_popt[0], sigmoid_popt[1], sigmoid_popt[2], sigmoid_popt[3]), 'r-', label=f'Fit: a = {sigmoid_popt[0]:.2e}, b = {sigmoid_popt[1]:.2e}, c = {sigmoid_popt[2]:.2e}, d = {sigmoid_popt[3]:.2e}')
    plt.title('Data Size vs Time per Access')
    plt.xlabel('Data Size')
    plt.ylabel('Time per Access')
    plt.legend()
    plt.grid(True)
    plt.savefig(time_per_access_filename + '.png', dpi=100)
    plt.show()


if __name__ == '__main__':
    main()