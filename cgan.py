import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
#from torchvision.utils import save_image

#from torch.utils.data import DataLoader
#from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("generated traces")

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=256, help="number of classes for dataset")         # number of classes = number of possible subkeys
#parser.add_argument("--sample_interval", type=int, default=400, help="interval between each sample")   # DA DEFINIRE
parser.add_argument("--trace_size_x", type=int, default=80000, help="number of elements on the x axes (n. samples)")
parser.add_argument("--trace_size_y", type=int, default=20000, help="number of elements on the y axes (n. data)")
#parser.add_argument("--channels", type=int, default=1, help="number of image channels")

opt = parser.parse_args()
print(opt)

trace_shape = (opt.trace_size_x, opt.trace_size_y)

cuda = True if torch.cuda.is_available() else False

# ----------------
# Load the dataset
# ----------------

path = '/Users/magdalenasolitro/Desktop/AI&CS MSc. UniUD/Small Project in CS/first order masked AES-128 2 rounds 2/'

traces = None

for file in os.listdir(path + 'traces'):       # array di stringhe, ogni elemento è il nome di un file
    open_file = open(path + 'traces/' + file, 'rb')
    #var = get_var(file)
    if 'random_traces' in file and not 'test' in file:
        if traces is None:      # viene eseguito quando non abbiamo ancora aperto nessun file (prima iterazione)
            traces = np.array(pickle.load(open_file))
        else:
            trace_temp = np.array(pickle.load(open_file))   # current file
            # ogni elem. di traces contiene un file (che rappresenta la pwr trace)
            traces = np.append(traces,trace_temp,axis=0)

# load the labels
labels = np.load(path + 'labels/s1.npy')


# ----------
# Generator
# ----------

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # input: class label, output: vector embedding for the label
        # 1st arg: size of the dictionary of embeddings (n. rows)
        # 2nd arg: size of each embedding vector (n. columns, = n. features)
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        # one block is composed by a Linear unit, followed by a LeakyReLU activation function
        def block(in_feat, out_feat, normalize=True):
            # Applies a linear transformation to the incoming data
            layers = [nn.Linear(in_feat, out_feat)]     # in_feat = size of each input sample
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # 0.2 = angle of the negative slope
            return layers

        # model definition
        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(trace_shape))),     # npp.prod() returns the product of array elements over a given axis.
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and noise to produce input of the generator
        # torch.cat() concatenates the given sequence of tensors in the given dimension
        # -1 is the dimension over which the tensors are concatenated
        gen_input = torch.cat((self.label_emb(labels), noise), -1)

        # Generate a trace out of gen_input, passing it through the model
        trace = self.model(gen_input)

        # Reshape the tensor according to trace_shape
        # trace.size(0) returns an int representing the size of the trace tensor in the 0th dim
        # trace = trace.view(trace.size(0), *trace_shape)
        trace = trace.view(opt.trace_size_x)

        # ora trace ha la forma desiderata: è un tensore di una riga, con 80000 colonne (i samples)

        return trace


# -------------
# Discriminator
# -------------

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(trace_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, trace, labels):
        # Concatenate label embedding and trace to produce input
        d_in = torch.cat((trace, self.label_embedding(labels)), -1)
        classify = self.model(d_in)

        return classify


# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


# DA RIVEDERE
def sample_trace(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""

    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))

    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_trs = generator(z, labels)
    #save_trace(gen_trs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for trs in traces:

        # Adversarial ground truths                                 # gradients do not need to be computed for this Tensor
        valid = Variable(FloatTensor(opt.batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(opt.batch_size, 1).fill_(0.0), requires_grad=False)

        # FloatTensor(opt.batch_size, 1).fill_(1.0) = tensore di float, batch_size righe e 1 colonna, riempite con 1.0

        # Configure input
        real_trs = Variable(trs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        # np.random.normal = draw random samples from a normal (Gaussian) distribution. Mean = 0, std_dev = 1 (normalization)
        # output shape = (opt.batch_size, opt.latent_dim)
        z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

        # np.random.randint = return random integers from 0 (inclusive) to n_classes (exclusive).
        # output shape = batch_size
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, opt.batch_size)))

        # Generate a batch of traces
        gen_trs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_trs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_trs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_trs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
