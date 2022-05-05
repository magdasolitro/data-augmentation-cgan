# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 10:08:34 2022

@author: martho
"""

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    images = np.load('metrics/images.npy')
    print(images.shape)
    plt.plot(images[0])
    plt.show()

    # retrieve significant trace window around the first timepoint
    # time_points = np.load('/Users/magdalenasolitro/Desktop/AI&CS MSc. UniUD/Small Project in '
    #                  'CS/dataset_joey/timepoints/s.npy')
    #
    # # time point in the first round in which AES applies SubBytes to the first byte of the state
    # this_timepoint = time_points[0]
    #
    # # select window around the point
    # start = this_timepoint - 1000
    # end = this_timepoint + 1000
    #
    # images = np.load('/Users/magdalenasolitro/Desktop/AI&CS MSc. UniUD/Small Project in '
    #                  'CS/dataset_joey/tracedata/random_keys_traces_0.npy')
    # print(images.shape)
    # plt.plot(images[0][start:end])
    # plt.show()
