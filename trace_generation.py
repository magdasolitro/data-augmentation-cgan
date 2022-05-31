import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test gans')

    parser.add_argument('--SAVE', action="store_true", dest="SAVE",
                        help='Save fids', default=False)
    parser.add_argument('-n', action="store", dest="N", help='Number of real traces compared too',
                        type=int, default=100)    
    
    args = parser.parse_args()
    N = args.N
    SAVE = args.SAVE    

    model = load_model('models/' + 's003_gan_generator_661.h5')
    
    n_classes = 256
    
    images = None
    labels = None
    for iterations in range(N//1000):
        label = np.random.randint(0, high=n_classes - 1, size=1000)
        noise = tf.random.normal([1000, 100])
        image = np.array(model([noise, label], training=False))
        image = image.reshape(-1, 1000)
        images = image if images is None else np.append(images, image, axis=0)
        labels = label if labels is None else np.append(labels, label, axis=0)
    print(images.shape)
    print(labels.shape)

    np.save('images.npy', images, allow_pickl=True)
    np.save('labels.npy', labels, allow_pickle=True)

