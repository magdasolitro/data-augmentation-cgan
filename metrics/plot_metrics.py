# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 09:41:26 2021

@author: martho
"""

import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
#from scipy.linalg import sqrtm
import os
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from gan import make_generator_model,make_discriminator_model
# from utility import *
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
    metrics = pd.read_csv('metrics.csv', usecols = ['gen_loss','real_acc','fake_acc','fake_loss','real_loss'])

    d = string_to_dict()
    df = pd.DataFrame(data=d)
    print(df)

    metrics['MA_fake_acc'] = df['fake_acc'].rolling(window = 100).mean()
    metrics['MA_real_acc'] = df['real_acc'].rolling(window = 100).mean()

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


if __name__ == "__main__":
    plot_accuracy_loss()


# def plot_image():
#     #fake_image = np.load("generated_image.npy")
#     distorsion = 'ma'
#     train_data = load_dataset_gan(distorsion)
#     real_image = reshape_array(train_data[5])
#     #plt.plot(fake_image)
#     plt.plot(real_image)
#
#
#
# def calculate_fid(act1, act2):
# 	# calculate mean and covariance statistics
# 	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
# 	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
# 	# calculate sum squared difference between means
# 	ssdiff = numpy.sum((mu1 - mu2)**2.0)
# 	# calculate sqrt of product between cov
# 	covmean = sqrtm(sigma1.dot(sigma2))
# 	# check and correct imaginary numbers from sqrt
# 	if iscomplexobj(covmean):
# 		covmean = covmean.real
# 	# calculate score
# 	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
# 	return fid
#
#
# def assess_gan(distorsion):
#     fake_image = np.load("images/"+"generated_image_{}.npy".format(distorsion))
#     train_data = load_dataset_gan(distorsion)
#     real_image = reshape_array(train_data[0])
#     random_image = np.random.sample(10000).reshape(500,-1)
#     fid = calculate_fid(fake_image.reshape(500,-1),real_image.reshape(500,-1))
#     random_fid = calculate_fid(random_image,real_image.reshape(500,-1))
#     print(fid)
#     print(random_fid)




# generator = make_generator_model()
# discriminator = make_discriminator_model()
# generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
# checkpoint_dir = './training_checkpoints/'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt-5")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                   discriminator_optimizer=discriminator_optimizer,
#                                   generator=generator,
#                                   discriminator=discriminator)
# noise = tf.random.normal([1, 100])


# # # checkpoint.restore(checkpoint_prefix)

# generator = load_model(MODEL_FOLDER+'gan_generator_2708.h5')



# generated_image = generator(noise, training=False)
# reshaped_image= reshape_tensor(generated_image)
# np.save("generated_image.npy",reshaped_image)
# plt.plot(reshaped_image)

# distorsion = 'res'

# train_data = load_dataset_gan(distorsion)
