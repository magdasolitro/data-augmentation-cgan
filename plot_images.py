# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 10:08:34 2022

@author: martho
"""

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    images = np.load('metrics/images.npy')
    plt.plot(images[0])     # plots the first trace in the file (index 0)
    plt.show()

