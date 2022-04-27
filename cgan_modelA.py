# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 14:55:25 2021

@author: martho
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import argparse
import time
import sys

from tensorflow.keras import backend as K

# Batch and shuffle the data

if sys.platform == 'win32':
    DATASET_FOLDER_FURIOUS = 'D:/dataset_joey/'
else:
    DATASET_FOLDER_FURIOUS = '/home/solitroma/Desktop/small project/dataset_joey/'

DATASET_FOLDER_FURIOUS = 'D:/dataset_joey/' if sys.platform == 'win32' else '/srv/dataset_furious/'
#DATASET_FOLDER_FURIOUS = '/home/solitroma/Desktop/small project/dataset_joey/'

PROJECT_FOLDER = 'C:/Users/martho/Documents/data-augmentation-cgan/' if sys.platform == 'win32' else '/home/martho/Projets/data-augmentation-cgan/'
#PROJECT_FOLDER = '/home/solitroma/Desktop/small project/data-augmentation-cgan/'

METRICS_FOLDER = PROJECT_FOLDER + 'metrics/'
MODEL_FOLDER = PROJECT_FOLDER + 'models/'

TRACES_FOLDER_FURIOUS = DATASET_FOLDER_FURIOUS + 'tracedata/'
REALVALUES_FOLDER_FURIOUS = DATASET_FOLDER_FURIOUS + 'realvalues/'
TIMEPOINTS_FOLDER_FURIOUS = DATASET_FOLDER_FURIOUS + 'timepoints/'
POWERVALUES_FOLDER_FURIOUS = DATASET_FOLDER_FURIOUS + 'powervalues/'

BUFFER_SIZE = 60000
BATCH_SIZE = 200

INTERMEDIATES = ['s1']
VARIABLE_LIST = {}
for intermediate in INTERMEDIATES:
    VARIABLE_LIST[intermediate] = [
        intermediate[:len(list(intermediate)) - 1] + '0' + ('0' + str(i) if i < 10 else '' + str(i)) for i in
        range(1 if int(intermediate[len(list(intermediate)) - 1]) == 1 else 17,
              17 if int(intermediate[len(list(intermediate)) - 1]) == 1 else 33)]


def make_generator_model(n_classes = 256,embedding_dim = 100):
    
    
    noise_input = layers.Input(shape = (100,))



    reshaped = layers.Reshape((1,100))(noise_input)
    label_input = layers.Input(shape = (1,))
    embedded = layers.Embedding(n_classes, embedding_dim)(label_input)

    concat_input = layers.Concatenate()([reshaped, embedded])
  
    x = layers.Dense(500)(concat_input)
    x = layers.Reshape((500,1))(x)
    x = layers.Conv1DTranspose(500, 5, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv1DTranspose(250, 5, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv1DTranspose(100, 5, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv1DTranspose(50, 5, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.AveragePooling1D(1, strides=10)(x)

    output = layers.Conv1DTranspose(20, 5, strides=1, padding='same', use_bias=False, activation='sigmoid')(x)
 
    model = Model(inputs = [noise_input,label_input],outputs = [output])
    model.summary()
    return model




def make_discriminator_model(n_classes = 256,embedding_dim = 8):

    image_input = layers.Input(shape = (50,20))
    label_input = layers.Input(shape = (1,))
    embedded = layers.Embedding(n_classes, embedding_dim)(label_input)
    dense = layers.Dense(1000)(embedded)
    reshaped  = layers.Reshape((50,20))(dense)
    concat_input = layers.Concatenate()([image_input, reshaped])
    

    x = layers.Conv1D(32, 5, strides=2, padding='same')(concat_input)
    x = layers.Lambda(lambda x: K.l2_normalize(x,axis=1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv1D(64,5, strides=2, padding='same')(x)
    x = layers.Lambda(lambda x: K.l2_normalize(x,axis=1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv1D(128,5, strides=2, padding='same')(x)
    x = layers.Lambda(lambda x: K.l2_normalize(x,axis=1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.AveragePooling1D(2, strides=2)(x)


    x = layers.Flatten()(x)
    x = layers.Dense(50, activation='relu')(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dense(50, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs = [image_input,label_input],outputs = [output])

    return model


def discriminator_loss(real_output, fake_output):
    real_loss = bin_cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = bin_cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss, fake_loss, real_loss


def discriminator_accuracy(real_output, fake_output):
    real_accuracy = tf.reduce_sum(tf.where(real_output >= 0.5, tf.ones_like(real_output), tf.zeros_like(real_output)))
    fake_accuracy = tf.reduce_sum(tf.where(fake_output >= 0.5, tf.zeros_like(fake_output), tf.ones_like(fake_output)))
    return 2*fake_accuracy / (fake_output.shape[0]), 2*real_accuracy/ (real_output.shape[0])


def generator_loss(fake_output):
    return bin_cross_entropy(tf.ones_like(fake_output), fake_output)


def load_furious_traces(n, timepoint, window):
    traces = None

    for i in range(20):
        file = TRACES_FOLDER_FURIOUS + ('random_keys_traces_{}'.format(i)) + '.npy'
        print('Loading {}'.format(file))
        traces_full = np.load(file, allow_pickle=True)[:, timepoint - window // 2: timepoint + window // 2]

        if traces is None:
            traces = traces_full
        else:
            traces = np.append(traces, traces_full, axis=0)
        if len(traces) > n:
            return reshaped_gan(traces[:n])
    return reshaped_gan(traces)


def normalise_neural_trace(v):
    # Shift up
    return v - np.min(v)


def normalise_neural_trace_single(v):
    return divide_rows_by_max(normalise_neural_trace(v))


def divide_rows_by_max(X):
    if len(X.shape) == 1:
        return X.astype(np.float32) / np.max(X)
    else:
        return X.astype(np.float32) / np.max(X, axis=1)[:, None]


def normalise_neural_traces(X):
    if X.shape[0] > 200000:
        # Memory error: do sequentially
        out = np.empty(X.shape)
        for i in range(X.shape[0]):
            out[i] = normalise_neural_trace_single(X[i])
        return out
    else:
        # DEBUG
        minimum_value_zero = np.apply_along_axis(normalise_neural_trace, 1, X)
        divided_by_max = divide_rows_by_max(minimum_value_zero)
        return divided_by_max


def reshaped_gan(dataset):
    print(dataset.shape)
    normalised_traces = normalise_neural_traces(dataset)

    return normalised_traces.reshape((normalised_traces.shape[0], 50, 20))


def load_dataset_gan(n_traces=200000, variable=None, training=True, window=1000):
    values = np.load(REALVALUES_FOLDER_FURIOUS + 's' + '.npy')[VARIABLE_LIST['s1'].index(variable), :]
    timepoint = np.load(TIMEPOINTS_FOLDER_FURIOUS + 's' + '.npy')[VARIABLE_LIST['s1'].index(variable)]
    traces = load_furious_traces(n_traces, timepoint, window)

    idx = np.random.permutation(traces.shape[0])
    traces = traces[idx]
    real_values = values[idx]
    return traces, real_values


@tf.function
def train_step(images, target):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, target], training=True)

        real_output = discriminator([images, target], training=True)
        fake_output = discriminator([generated_images, target], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss, fake_loss, real_loss = discriminator_loss(real_output, fake_output)
        fake_accuracy, real_accuracy = discriminator_accuracy(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return disc_loss, gen_loss, fake_loss, real_loss, fake_accuracy, real_accuracy


def train(dataset, epochs, dataset_size, var,bs):
    loss_real_dict = {}
    loss_fake_dict = {}
    loss_gen_dict = {}
    acc_real_dict = {}
    acc_fake_dict = {}
    best = 10

    for epoch in range(epochs):
        start = time.time()
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_f_loss = 0
        epoch_r_loss = 0
        epoch_f_acc = 0
        epoch_r_acc = 0
        for image_batch, target_batch in dataset:
            d_loss, g_loss, f_loss, r_loss, f_accuracy, r_accuracy = train_step(image_batch, target_batch)
            epoch_d_loss += d_loss
            epoch_g_loss += g_loss
            epoch_f_loss += f_loss
            epoch_r_loss += r_loss
            epoch_f_acc += f_accuracy
            epoch_r_acc += r_accuracy
        epoch_d_loss /= (dataset_size / bs)
        epoch_g_loss /= (dataset_size / bs)
        epoch_f_loss /= (dataset_size / bs)
        epoch_r_loss /= (dataset_size / bs)
        epoch_f_acc /= (dataset_size / bs)
        epoch_r_acc /= (dataset_size / bs)
        # Save the model every 15 epochs
        loss_real_dict[epoch] = epoch_r_loss
        loss_fake_dict[epoch] = epoch_f_loss
        loss_gen_dict[epoch] = epoch_g_loss
        acc_real_dict[epoch] = epoch_r_acc
        acc_fake_dict[epoch] = epoch_f_acc
        print('=========== EPOCH {} ==========='.format(epoch + 1))
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        print('Discriminator Loss is {}, Generator Loss is {} '.format(epoch_d_loss, epoch_g_loss))
        print('Fake loss is {}, Real loss is {} '.format(epoch_f_loss, epoch_r_loss))
        print('Fake accuracy is {}, Real accuracy is {} '.format(epoch_f_acc, epoch_r_acc))

        if (abs(epoch_r_acc - 50) + abs(epoch_f_acc - 50) <= best) and epoch > 200:
            print('Saved')
            generator.save(MODEL_FOLDER + '{}_gan_generator_{}.h5'.format(var if not var is None else 'all', epoch + 1))
            discriminator.save(
                MODEL_FOLDER + '{}_gan_discriminator_{}.h5'.format(var if not var is None else 'all', epoch + 1))
            best = abs(epoch_r_acc - 50) + abs(epoch_f_acc - 50)
            if best < 1:
                print('Early break!')
                return loss_real_dict, loss_fake_dict, loss_gen_dict, acc_real_dict, acc_fake_dict
    return loss_real_dict, loss_fake_dict, loss_gen_dict, acc_real_dict, acc_fake_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--COMBINE', action="store_true", dest="COMBINE",
                        help='Load all bytes of the intermediate', default=False)
    parser.add_argument('-e', '-epochs', action="store", dest="EPOCHS",
                        help='Number of Epochs in Training (default: 75 CNN, 100 MLP)',
                        type=int, default=100)
    parser.add_argument('-b', '-batch', '-batch_size', action="store", dest="BATCH_SIZE",
                        help='Size of Training Batch (default: 200)',
                        type=int, default=200)

    parser.add_argument('-v', '-variable', action="store", dest="VARIABLE", help='Variable chosen for the training',
                        type=str, default='s003')

    # Target node here
    args = parser.parse_args()
    EPOCHS = args.EPOCHS
    BATCH_SIZE = args.BATCH_SIZE
    COMBINE = args.COMBINE
    VARIABLE = args.VARIABLE

    generator = make_generator_model()

    discriminator = make_discriminator_model()

    cross_entropy = tf.keras.losses.CategoricalCrossentropy()
    bin_cross_entropy = tf.keras.losses.BinaryCrossentropy()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    noise_dim = 100
    num_examples_to_generate = 16

    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    train_data, real_values = load_dataset_gan(variable=VARIABLE)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, real_values)).shuffle(train_data.shape[0]).batch(
        BATCH_SIZE)
    rl, fl, gl, ra, fa = train(train_dataset, EPOCHS, train_data.shape[0], VARIABLE,BATCH_SIZE)
    metrics = pd.DataFrame.from_dict(rl, columns=['real_loss'], orient='index')
    metrics.insert(1, 'fake_loss', fl.values(), True)
    metrics.insert(1, 'fake_acc', fa.values(), True)
    metrics.insert(1, 'real_acc', ra.values(), True)
    metrics.insert(1, 'gen_loss', gl.values(), True)
    metrics.to_csv('metrics.csv', columns=['gen_loss', 'real_acc', 'fake_acc', 'fake_loss', 'real_loss'])
