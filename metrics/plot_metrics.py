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
from scalib.metrics import SNR

import pandas as pd
import matplotlib.pyplot as plt
import csv


# gather values for gen_loss
gen_loss = []
real_acc = []
fake_acc = []
fake_loss = []
real_loss = []

REAL_TRS_PATH = '/home/solitroma/Desktop/small project/dataset_joey/tracedata/random_keys_traces_5.npy'
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
def normalise_trace(trace, num_samples=1000, min=-10000, max=20000):
    for val in range(num_samples):
        trace[val] = (trace[val] * (max-min)) + min

    return trace


def calculate_snr(fake_trs_path, fake_labels_path, real_trs_path, real_labels_path):
    # load fake traces
    fake_traces = np.load(fake_trs_path)

    # normalise fake traces between -10000 and 20000 (range of values of a real trace)
    for trace in range(fake_traces.shape[0]):
        fake_traces[trace] = normalise_trace(fake_traces[trace])

    # type conversion required for snr computation
    fake_traces = fake_traces.astype(np.int16)

    fake_labels = np.load(fake_labels_path)         # load the labels
    fake_labels = fake_labels.astype(np.uint16)     # type conversion required for snr computation
    fake_labels = np.reshape(fake_labels, (10000, 1))

    # compute snr values
    snr_fake = SNR(255, 1000)
    snr_fake.fit_u(fake_traces, fake_labels)
    snr_val_fake = snr_fake.get_snr()
    snr_val_fake_mean = np.mean(snr_val_fake, axis=0)

    # load real traces
    real_traces = np.load(real_trs_path)
    start, end = retrieve_window()
    real_traces = real_traces[:, start:end]
    real_traces = real_traces.astype(np.int16)

    real_labels = np.load(real_labels_path)
    real_labels = real_labels[0, :10000]    # only consider the first intermediate result for the first 10000 traces!
    real_labels = np.reshape(real_labels, (10000, 1))
    real_labels = real_labels.astype(np.uint16)

    snr_real = SNR(255, 1000)
    snr_real.fit_u(real_traces, real_labels)
    snr_val_real = snr_real.get_snr()

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].plot(snr_val_fake[0])
    axs[0].set_title('SNR (fake traces)')
    axs[0].set_xlabel('Sample')
    axs[0].set_ylabel('SNR')
    fig.suptitle('SNR', fontsize=16)

    axs[1].plot(snr_val_real[0])
    axs[1].set_title('SNR (real traces)')
    axs[1].set_xlabel('Sample')
    axs[1].set_ylabel('SNR')

    plt.show()


if __name__ == "__main__":
    # plot accuracy and loss
    plot_accuracy_loss()

    # compute Signal-to-Noise ratio
    # calculate_snr(FAKE_TRS_PATH, FAKE_LB_PATH, REAL_TRS_PATH, REAL_LB_PATH)

    # compute FID score
    # start, end = retrieve_window()
    start, end = 8100, 9100

    all_real_traces = np.load(REAL_TRS_PATH)
    real_traces_mean = np.mean(all_real_traces, axis=0)
    real_traces_mean = real_traces_mean[start:end]

    fake_traces = np.load(FAKE_TRS_PATH)
    # compute the mean of all traces
    fake_traces_mean = np.mean(fake_traces, axis=0)
    fake_traces_mean = normalise_trace(fake_traces_mean)

    # Plot real and fake traces
    # fig, axs = plt.subplots(2, 1)
    # axs[1].plot(fake_traces_mean)
    # axs[1].set_title('Fake traces (mean)')
    # axs[1].set_xlabel('Time points')
    # axs[1].set_ylabel('Power value')
    #
    # axs[0].plot(real_traces_mean)
    # axs[0].set_title('Real traces (mean)')
    # axs[0].set_xlabel('Time points')
    # axs[0].set_ylabel('Power value')
    #
    # plt.subplots_adjust(hspace=0.7)
    #
    # plt.show()

    fid1 = calculate_fid(fake_traces_mean.reshape(50, 20), real_traces_mean.reshape(50, 20))
    fid2 = calculate_fid(fake_traces_mean.reshape(50, 20), fake_traces_mean.reshape(50, 20))

    print(fid1)
    print(fid2)

