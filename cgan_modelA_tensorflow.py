n_epochs = 200      # number of epochs of training
batch_size = 64     # size of the batches
lr = 0.0002         # adam: learning rate
b1 = 0.5            # adam: decay of first order momentum of gradient
b2 = 0.999          # adam: decay of first order momentum of gradient
n_cpu = 8           # number of cpu threads to use during batch generation
latent_dim = 100    # dimensionality of the latent space
n_classes = 256     # number of classes for dataset
target_op = 's'     # the intermediate result we want to attack
wnd_size = 500      # window size around a time point


def generator(n_classes=n_classes, embedding_dim=latent_dim):
    noise = layers.Input(shape = (100,))
    reshaped = layers.Reshape((1,100))(noise)

    label = layers.Input(shape=(1,))
    embedded = layers.Embedding(n_classes, embedding_dim)(label)

    input = layers.Concatenate()([reshaped, embedded])

    x = layers.Dense(100)(input)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(250)(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(500)(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    output = layers.Conv1D(filters=20, kernel_size=5, padding='same', use_bias=False)(x)

    model = Model(inputs = [noise, label], outputs = [output])

    return model


def discriminator(n_classes=n_classes, latent_dim=8):
    trace = layers.Input(shape=(1,500))     # TO BE OPTIONALLY MODIFIED
    label = layers.Input(shape=(1,))

    embedded_label = layers.Embedding(n_classes, latent_dim)(label)

    input = layers.Concatenate()([trace, embedded_label])

    x = layers.Dense(512)(input)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dense(512)(input)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(512)(input)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)

    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs = [trace, label], output = [output])

    return model
