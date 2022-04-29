import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Power Traces')
    parser.add_argument('-m', action='store', type=Model, nargs=1, help='generated model', metavar='M', dest='MODEL')

    args = parser.parse_args()
    model = args.MODEL
    n_classes = 256

    label = np.random.randint(0, n_classes - 1)
    noise = tf.random.normal([1, 100])

    image = model([noise, label], training=False)
    image = tf.reshape(image, (1000,))

    plt.title("Generated power trace")
    plt.ylabel("Power values")
    plt.plot(image)
    plt.show()
