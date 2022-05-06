# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 09:41:26 2021

@author: martho
"""

import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
import scipy
from scipy.linalg import sqrtm
import os
from scalib.metrics import SNR
import sys

import pandas as pd
import matplotlib.pyplot as plt
import csv


# gather values for gen_loss
gen_loss = []
real_acc = []
fake_acc = []
fake_loss = []
real_loss = []

def retrieve_window():
    # retrieve significant trace window around the first timepoint
    time_points = np.load('/home/solitroma/Desktop/small project/dataset_joey/timepoints/s.npy')

    # time point in the first round in which AES applies SubBytes to the first byte of the state
    this_timepoint = time_points[0]

    # select window around the point
    start = this_timepoint - 500
    end = this_timepoint + 500

    return start, end

def string_to_dict():
    file = open('metrics.csv')
    csvreader = csv.reader(file)

    header = []
    header = next(csvreader)

    for row in csvreader:
        tensor = row[1]
        tmp = tensor[10:]
        tensor_value = tmp.split(",")[0]
        gen_loss.append(tensor_value)

        tensor = row[2]
        tmp = tensor[10:]
        tensor_value = tmp.split(",")[0]
        real_acc.append(tensor_value)

        tensor = row[3]
        tmp = tensor[10:]
        tensor_value = tmp.split(",")[0]
        fake_acc.append(tensor_value)

        tensor = row[4]
        tmp = tensor[10:]
        tensor_value = tmp.split(",")[0]
        fake_loss.append(tensor_value)

        tensor = row[5]
        tmp = tensor[10:]
        tensor_value = tmp.split(",")[0]
        real_loss.append(tensor_value)

    # from float to dictionary
    dict = {'gen_loss': gen_loss,
            'real_acc': real_acc,
            'fake_acc': fake_acc,
            'fake_loss': fake_loss,
            'real_loss': real_loss}

    return dict


def plot_accuracy_loss():
    metrics = pd.read_csv('metrics.csv', usecols=['gen_loss', 'real_acc', 'fake_acc', 'fake_loss', 'real_loss'])

    d = string_to_dict()
    df = pd.DataFrame(data=d)

    metrics['MA_fake_acc'] = df['fake_acc'].rolling(window=100).mean()
    metrics['MA_real_acc'] = df['real_acc'].rolling(window=100).mean()

    df = df.astype(float)

    plot1 = plt.figure(1)
    ax = plt.gca()
    metrics.plot(y='MA_fake_acc', ax=ax)
    metrics.plot(y='MA_real_acc', ax=ax)

    plot2 = plt.figure(2)
    ax = plt.gca()
    df.plot(y='gen_loss', ax=ax)
    df.plot(y='fake_loss', ax=ax)
    df.plot(y='real_loss', ax=ax)

    plt.show()

    #gl.plot(ax = ax)


# def plot_image():
#     #fake_image = np.load("generated_image.npy")
#     distorsion = 'ma'
#     train_data = load_dataset_gan(distorsion)
#     real_image = reshape_array(train_data[5])
#     #plt.plot(fake_image)
#     plt.plot(real_image)


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)

    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
    	covmean = covmean.real

    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid

# def assess_gan(distorsion):
#     fake_image = np.load("images/"+"generated_image_{}.npy".format(distorsion))
#     train_data = load_dataset_gan(distorsion)
#     real_image = reshape_array(train_data[0])
#     random_image = np.random.sample(10000).reshape(500,-1)
#     fid = calculate_fid(fake_image.reshape(500,-1),real_image.reshape(500,-1))
#     random_fid = calculate_fid(random_image,real_image.reshape(500,-1))
#     print(fid)
#     print(random_fid)


def normalise_trace(trace, num_samples=1000, min=-10000, max=20000):
    for val in range(num_samples):
        trace[val] = (trace[val] * (max-min)) + min

    return trace


if __name__ == "__main__":
    # plot accuracy and loss
    plot_accuracy_loss()

    # compute Signal-To-Noise ratio
    fake_traces = np.load('images.npy')

    for trace in range(fake_traces.shape[0]):
        fake_traces[trace] = normalise_trace(fake_traces[trace])
    fake_traces = fake_traces.astype(np.int16)

    # plot1 = plt.figure(1)
    # for i in range(10):
    #     plt.plot(fake_traces[i])

    labels = np.load('labels.npy')
    labels = labels.astype(np.uint16)
    labels = np.reshape(labels, (10000, 1))

    snr = SNR(255, 1000)
    snr.fit_u(fake_traces, labels)
    snr_val = snr.get_snr()

    # plot2 = plt.figure(2)
    # plt.plot(snr_val[0])
    # plt.show()

    # compute FID score
    start, end = retrieve_window()
    all_real_traces = np.load('/home/solitroma/Desktop/small project/dataset_joey/tracedata/random_keys_traces_0.npy')
    real_trace = all_real_traces[0][start:end]
    fake_trace = fake_traces[0]
    fid = calculate_fid(fake_trace, real_trace)
    print(fid)
