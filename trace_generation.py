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
    
    args            = parser.parse_args()
    N = args.N
    SAVE = args.SAVE    

    model = load_model('models/' + 's003_gan_generator_661.h5')
    
    n_classes = 256
    
    label = np.random.randint(0, high  = n_classes - 1,size = N)
    noise = tf.random.normal([N, 100])

    image = np.array(model([noise, label], training=False))
    image = image.reshape(-1,1000)
    np.save('images.npy',image,allow_pickle= True ) 
    np.save('labels.npy',label,allow_pickle= True)


