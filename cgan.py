import argparse
import os

import numpy as np

import sys

if sys.platform == 'win32' or sys.platform == 'darwin':
    import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from tracedataset.TraceDataset import TraceDataset

os.makedirs("generated traces", exist_ok=True)

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=256, help="number of classes for dataset")
parser.add_argument("--target_op", type=str, default='s', help="the intermediate result we want to attack")
parser.add_argument("--wnd_size", type=int, default=560, help="window size around a time point")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between trace sampling")

opt = parser.parse_args()
print(opt)

trace_shape = (1, 2 * opt.wnd_size)

cuda = True if torch.cuda.is_available() else False

# ----------------
# Load the dataset
# ----------------

if sys.platform == 'win32' or sys.platform == 'darwin':
    path = '/Users/magdalenasolitro/Desktop/AI&CS MSc. UniUD/Small Project in CS/dataset_joey/'
else:
    path = '/media/usb/MIG/2.0 TB Volume/dataset_furious/'

# retrieve significant trace window around the first timepoint
time_points = np.load(path + 'timepoints/' + opt.target_op + '.npy')

# time point in the first round in which AES applies SubBytes to the first byte of the state
this_timepoint = time_points[0]

# select window around the point
start = this_timepoint - opt.wnd_size
end = this_timepoint + opt.wnd_size

# auxiliary data structures for trace processing
num_file = 0
trimmed_traces = None

# load all the files
print("Loading the dataset files...")
for i in range(20):
    file = path + 'tracedata/' + ('random_keys_traces_{}'.format(i)) + '.npy'
    if 'fixed' not in file and '.DS_Store' not in file:

        print("Processing file " + file + '...', end=' ')
        traces = np.load(file, allow_pickle=True)

        if trimmed_traces is None:
            trimmed_traces = traces[:, start:end]
        else:
            trimmed_traces = np.append(trimmed_traces, traces[:, start:end], axis=0)
        print('Done!')
    num_file += 1

print("Done!")

# ------------
# Training Set
# ------------
print("Creating the training set...")

# Compute number of total traces
n_data = trimmed_traces.shape[0]

labels_dict = dict()  # dictionary that associates
idx = np.arange(n_data)  # list of indexes for the dataset

# Load the labels
labels = np.load(path + 'realvalues/' + opt.target_op + '.npy')

# Associate IDs to labels
for i in range(n_data):
    new_dict = {idx[i]: labels[:, i]}
    labels_dict.update(new_dict)

training_set = TraceDataset(idx, labels_dict, trimmed_traces)

print('Done!')

# Set dataloader parameters
dataloader = torch.utils.data.DataLoader(
    training_set,
    batch_size=opt.batch_size,
    shuffle=True,
)


# Reshape wrapper
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


# Random initialization of weights
def weights_init_normal(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        torch.nn.init.normal_(m.weight)


# ----------
# Generator
# ----------

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 1 arg = size of the dictionary of embeddings, 2 arg = the size of each embedding vector
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + opt.latent_dim, 500),
            Reshape((opt.batch_size, 1, 500)),  # (batch size, n_channels, signal_length)
            nn.ConvTranspose1d(1, 500, (5,), bias=False),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(500, 250, (5,), bias=False),
            nn.BatchNorm1d(250),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(250, 100, (5,), bias=False),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(100, 50, (5,), bias=False),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(),

            nn.AvgPool1d(1, stride=10),
            nn.ConvTranspose1d(50, 20, (5,), bias=False),

            nn.Flatten(),
            Reshape((opt.batch_size, 1, -1)),
            nn.Sigmoid()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and noise to produce input of the generator
        gen_input = torch.cat((self.label_emb(labels), noise), -1)

        # Generate a trace out of gen_input, passing it through the model
        trace = self.model(gen_input)

        return trace


# -------------
# Discriminator
# -------------

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Conv1d(1, 32, (5,), stride=(2,), bias=False),
            nn.Flatten(),
            nn.Linear(21952, 1),
            nn.Sigmoid()
        )

    def forward(self, trace, labels):
        # Concatenate label embedding and trace to produce input
        labels = self.label_emb(labels)
        labels = labels.view(opt.batch_size, 1, -1)

        d_in = torch.cat((trace, labels), -1)
        classify = self.model(d_in)

        return classify


# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

if cuda:
    device = torch.device("cuda")
    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)
else:
    device = torch.device("cpu")

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------

# Keep track of the loss
d_loss_list, g_loss_list = [], []

for epoch in range(opt.n_epochs):
    for i, (real_trs, labels) in enumerate(dataloader):
        real_trs = real_trs.type(LongTensor)
        labels = labels.type(LongTensor)

        # Ground truths
        valid = FloatTensor(opt.batch_size, 1).fill_(1.0)
        fake = FloatTensor(opt.batch_size, 1).fill_(0.0)

        real_trs = real_trs.view((opt.batch_size, 1, 2 * opt.wnd_size))

        # Generate fake traces
        z = FloatTensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim)))
        gen_trs = generator(z, labels)

        # -------------------
        # Train Discriminator
        # -------------------

        optimizer_D.zero_grad()

        # Evaluate real traces
        classify_real = discriminator(real_trs, labels)
        d_real_loss = adversarial_loss(classify_real, valid)

        # Evaluate generated traces
        classify_fake = discriminator(gen_trs.detach(), labels)
        d_fake_loss = adversarial_loss(classify_fake, fake)

        # Compute total loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss_list.append(d_loss)

        d_loss.backward()
        optimizer_D.step()

        # ---------------
        # Train Generator
        # ---------------

        optimizer_G.zero_grad()

        g_loss = adversarial_loss(discriminator(gen_trs, labels), valid)
        g_loss_list.append(g_loss)

        g_loss.backward()
        optimizer_G.step()

        # Plot the loss
        if sys.platform == "win32" or sys.platform == "darwin":
            if epoch == opt.n_epochs - 1 and i == opt.batch_size - 1:
                x = np.arange(0, opt.n_epochs)

                plot_gen = plt.figure(1)
                plt.title("Generator's loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.plot(x, g_loss_list)

                plot_disc = plt.figure(2)
                plt.title("Discriminator's loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.plot(x, d_loss_list)

                plt.show()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

np.save('Loss_gen', g_loss_list)
np.save('Loss_disc', d_loss_list)
