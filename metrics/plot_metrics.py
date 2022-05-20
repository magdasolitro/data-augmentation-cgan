# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 09:41:26 2021

@author: martho
"""

import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj

from scipy.linalg import sqrtm

import pandas as pd
import matplotlib.pyplot as plt
import csv


# gather values for gen_loss
gen_loss = []
real_acc = []
fake_acc = []
fake_loss = []
real_loss = []

REAL_TRS_PATH = '/home/solitroma/Desktop/small project/dataset_joey/tracedata/random_keys_traces_0.npy'
REAL_LB_PATH = '/home/solitroma/Desktop/small project/dataset_joey/realvalues/s.npy'

FAKE_TRS_PATH = '/home/solitroma/Desktop/small project/data-augmentation-cgan/metrics/images.npy'
FAKE_LB_PATH = '/home/solitroma/Desktop/small project/data-augmentation-cgan/metrics/labels.npy'


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

    # header = []
    # header = next(csvreader)

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
    return {'gen_loss': gen_loss,
            'real_acc': real_acc,
            'fake_acc': fake_acc,
            'fake_loss': fake_loss,
            'real_loss': real_loss}


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


# maps fake trace's values in the range [-10000, 20000], which is the range of real traces
def normalise_trace(trace, num_samples=1000, min=-6523, max=13789):
    for val in range(num_samples):
        trace[val] = (trace[val] * (max-min)) + min

    return trace


def compute_pearsoncorrelation(fake_trs_path, fake_labels_path, real_trs_path, real_labels_path):
    # load fake traces
    fake_traces = np.load(fake_trs_path)

    # normalise fake traces between -10000 and 20000 (range of values of a real trace)
    for trace in range(fake_traces.shape[0]):
        fake_traces[trace] = normalise_trace(fake_traces[trace])

    # type conversion required for snr computation
    fake_traces = fake_traces.astype(np.int16)
    fake_traces = np.swapaxes(fake_traces, 1, 0)
    fake_labels = np.load(fake_labels_path)         # load the labels
    fake_labels = fake_labels.astype(np.uint16)     # type conversion required for snr computation

    # Pearson's Correlation
    correlations_fake = np.array([np.abs(np.corrcoef(fake_labels, timeslice))[0, 1] for timeslice in fake_traces])

    # load real traces
    real_traces = np.load(real_trs_path)
    start, end = retrieve_window()
    real_traces = real_traces[:, start:end]
    real_traces = real_traces.astype(np.int16)
    real_traces = np.swapaxes(real_traces, 1, 0)

    real_labels = np.load(real_labels_path)
    real_labels = real_labels[0, :10000]    # only consider the first intermediate result for the first 10000 traces!
    real_labels = real_labels.astype(np.uint16)

    correlations_real = np.array([np.abs(np.corrcoef(real_labels, timeslice))[0, 1] for timeslice in real_traces])

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].plot(correlations_fake)
    axs[0].set_title("Pearson's Correlation (fake traces)")
    axs[0].set_xlabel('Sample')
    axs[0].set_ylabel('Correlation')

    axs[1].plot(correlations_real)
    axs[1].set_title("Pearson's Correlation (real traces)")
    axs[1].set_xlabel('Sample')
    axs[1].set_ylabel('Correlation')

    plt.show()


# Plot real and fake traces' mean values (for visual comparison)
def plot_traces(fake_trs_mean, real_trs_mean):
    fig, axs = plt.subplots(2, 1)
    axs[1].plot(fake_trs_mean)
    axs[1].set_title('Fake traces (mean)')
    axs[1].set_xlabel('Time points')
    axs[1].set_ylabel('Power value')

    axs[0].plot(real_trs_mean)
    axs[0].set_title('Real traces (mean)')
    axs[0].set_xlabel('Time points')
    axs[0].set_ylabel('Power value')

    plt.subplots_adjust(hspace=0.7)

    plt.show()

    # Plots fake and real traces together
    # plt.plot(fake_trs_mean)
    # plt.plot(real_trs_mean)
    #
    # plt.show()


if __name__ == "__main__":
    # plot accuracy and loss
    # plot_accuracy_loss()

    # compute Pearson's Correlation
    # compute_pearsoncorrelation(FAKE_TRS_PATH, FAKE_LB_PATH, REAL_TRS_PATH, REAL_LB_PATH)

    # compute FID score
    #start, end = retrieve_window()
    start, end = 8070, 9070

    all_real_traces = np.load(REAL_TRS_PATH)
    real_traces_mean = np.mean(all_real_traces, axis=0)
    real_traces_mean = real_traces_mean[start:end]

    # Retrieve maximum and minimum value in the portion of the traceTr
    # print('max: ' + str(np.amax(real_traces_mean)))
    # print('min: ' + str(np.amin(real_traces_mean)))

    fake_traces = np.load(FAKE_TRS_PATH)
    # compute the mean of all traces
    fake_traces_mean = np.mean(fake_traces, axis=0)
    fake_traces_mean = normalise_trace(fake_traces_mean)

    plot_traces(fake_traces_mean, real_traces_mean)

    fid1 = calculate_fid(fake_traces_mean.reshape(50, 20), real_traces_mean.reshape(50, 20))
    fid2 = calculate_fid(fake_traces_mean.reshape(50, 20), fake_traces_mean.reshape(50, 20))

    print(fid1)
    print(fid2)
