"""
Script that computes the Signal-To-Noise ratio of a generated trace
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    # da modificare con traccia generata
    # =========================================================================================================
    if sys.platform == 'darwin':
        path = '/Users/magdalenasolitro/Desktop/AI&CS MSc. UniUD/Small Project in CS/dataset_joey/'
    else:
        path = '/home/solitroma/Desktop/small project/dataset_joey/'

    time_points = np.load(path + 'timepoints/s.npy')
    wnd_size = 1000

    # time point in the first round in which AES applies SubBytes to the first byte of the state
    this_timepoint = time_points[0]

    # select window around the point
    start = this_timepoint - wnd_size
    end = this_timepoint + wnd_size

    # load all the files
    print("Loading the file...")
    file = path + 'tracedata/random_keys_traces_0.npy'
    trace = np.load(file, allow_pickle=True)
    trace = trace[:, start:end]
    signal = trace[0]
    print('Done.')
    # =========================================================================================================

    trs_length = trace.shape[1]
    snr = None
    diff = None
    tmp = 0
    max = 0
    min = 0
    for j in range(trs_length):     # loop over power values
        for i in range(10):         # loop over traces
            if i == 0:
                max = trace[i][j]
                min = trace[i][j]
            else:
                if trace[i][j] > max:
                    max = trace[i][j]
                elif trace[i][j] < min:
                    min = trace[i][j]
        tmp = abs(max - min)

        if diff is None:
            diff = tmp
        else:
            diff = np.append(diff, tmp)

    fig, axs = plt.subplots(1, 2)
    for i in range(10):
        axs[0].plot(trace[i])
    axs[0].set_title('10 different traces')
    axs[1].plot(diff)
    axs[1].set_title('SNR')

    plt.show()


