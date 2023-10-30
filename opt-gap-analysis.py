import re
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np

# Initialize variables to store data
timestamps = []
diffs = []
counters = []
Tspeedups = []
Fspeedups = []
O1_duration = []
O3_duration = []

filename = 'sawtooth_opt-gap-log'

# use speedup = (O3 speed - O1 speed) / O1 speed, where speed is 1 / running time.
# for matrix multiplication, we can use FLOPS.  For input size n,  FLOPS = 2n^3 / time.
# FLOPS as the speed

# Define a function to parse a block of data
def parse_block(block):
    parts = block[0].strip().split(', ')
    t1 = float(parts[0].split(': ')[1][:-1])  # Extract and convert t1 value
    t2 = float(parts[1].split(': ')[1][:-1])  # Extract and convert t2 value
    t3 = float(parts[2].split(': ')[1][:-1])  # Extract and convert t3 value
    diff = float(block[1].split(': ')[1])  # Extract and convert diff value
    counter = int(block[2].split(': ')[1])  # Extract and convert counter value

    sp1 = 1 / t1
    sp2 = 1 / t2
    sp3 = 1 / t3
    Tspeedup = (sp3 - sp1) / sp1
    print(f"Speedup using 1/running time: {Tspeedup}")

    # Calculate the number of FLOPs for each data point
    flops1 = 2 * counter ** 3 / t1
    flops2 = 2 * counter ** 3 / t2
    flops3 = 2 * counter ** 3 / t3
    Fspeedup = (flops3 - flops1) / flops1
    print(f"Speedup using FLOPS: {Fspeedup}")

    Tspeedups.append(Tspeedup)
    Fspeedups.append(Fspeedup)

    timestamps.append((t1, t2, t3))
    diffs.append(diff)
    counters.append(counter)
    O1_duration.append(t1)
    O3_duration.append(t3)


def main():
    with open(filename, 'r') as f:
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
    # print(timestamps)
    # print(diffs)
    # print(counters)

    # Plot the data
    plt.figure(figsize=(10, 6))
    # plt.plot(counters, Tspeedups, 'red', label='1/running time')
    plt.scatter(counters, Fspeedups, color='blue')
    # plt.plot(counters, Fspeedups, 'blue', label='FLOPS')
    plt.title('Size vs Speedups(FLOPS)')
    plt.xlabel('Size')
    plt.ylabel('Speedups(FLOPS)')
    plt.grid()
    plt.savefig(filename, dpi=100)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(counters, O1_duration, color='blue')
    plt.title('Size vs Duration (s) O1')
    plt.xlabel('Size')
    plt.ylabel('Duration (s)')
    plt.grid()
    plt.savefig(filename + '_O1', dpi=100)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(counters, O3_duration, color='blue')
    plt.title('Size vs Duration (s) O3')
    plt.xlabel('Size')
    plt.ylabel('Duration (s)')
    plt.grid()
    plt.savefig(filename + '_O3', dpi=100)
    plt.show()


if __name__ == '__main__':
    main()