import argparse
import os
import sys

import numpy as np

if sys.platform == 'win32' :
    import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from tracedataset.TraceDataset import TraceDataset

os.makedirs("generated traces", exist_ok=True)


# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=50, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=256, help="number of classes for dataset")
parser.add_argument("--target_op", type=str, default='s', help="the intermediate result we want to attack")
parser.add_argument("--wnd_size", type=int, default=280, help="window size around a time point")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between trace sampling")

opt = parser.parse_args()
print(opt)

trace_shape = (1, 2 * opt.wnd_size)

cuda = True if torch.cuda.is_available() else False


# ----------------
# Load the dataset
# ----------------

# path =  'D:/dataset_joey/' if sys.platform == 'win32' else '/media/usb/MIG/2.0 TB Volume/dataset_furious/'
path = '/Users/magdalenasolitro/Desktop/AI&CS MSc. UniUD/Small Project in CS/dataset_joey/' if sys.platform == 'win32'  else '/media/usb/MIG/2.0 TB Volume/dataset_furious/'

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

# Compute number of total traces and the number of samples
n_data = trimmed_traces.shape[0]
n_samples = trimmed_traces[1]

labels_dict = dict()            # dictionary that associates
idx = np.arange(n_data)         # list of indexes for the dataset

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


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


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
            Reshape((50, 1, 500)),      # (batch size, n_channels, signal_length)
            nn.ConvTranspose1d(1, 500, 5, bias=False),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(500, 250, 5, bias=False),
            nn.BatchNorm1d(250),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(250, 100, 5, bias=False),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(),

            nn.ConvTranspose1d(100, 50, 5, bias=False),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(),

            nn.AvgPool1d(1, stride=10),
            nn.ConvTranspose1d(50, 10, 5, bias=False),

            nn.Flatten(),
            Reshape((50, 1, -1)),
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

        self.embedding_dim = 8
        self.label_emb = nn.Embedding(opt.n_classes, self.embedding_dim)

        self.model = nn.Sequential(
            nn.Conv1d(1, 32, 5, stride=2, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32, 64, 5, stride=2, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 128, 5, stride=2, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.AvgPool1d(2, stride=2),
            nn.Flatten(),

            nn.Linear(12288, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward(self, trace, labels):
        # Concatenate label embedding and trace to produce input
        embedded = self.label_emb(labels)

        pre_processing = nn.Sequential(
            nn.Linear(8, 1000),
            Reshape((50, 1, 1000)),
        )

        output = pre_processing(embedded)
        d_in = torch.cat((trace, output), -1)
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

'''
def save_trace(trs, labels, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Create a file to store the generated traces
    filename1 = 'generated traces/gen_trace_' + str(batches_done) + '.npy'

    if filename1 in os.listdir('./generated traces/'):    # check existence of another file with identical name
        os.remove(filename1)                # if that's the case, remove file

    with open(filename1, 'x'):
        np.save(filename1, trs.detach())

    # Create a file for the labels
    filename2 = 'generated traces/labels_' + str(batches_done) + '.npy'

    if filename2 in os.listdir('./generated traces/'):
        os.remove(filename2)

    with open(filename2, 'x'):
        np.save(filename2, labels)
'''

# ----------
#  Training
# ----------

# Keep track of the loss
y_g = np.zeros(opt.n_epochs)
y_d = np.zeros(opt.n_epochs)

for epoch in range(opt.n_epochs):
    for i, (real_trs, labels) in enumerate(dataloader):
        # Adversarial ground truths
        valid = FloatTensor(opt.batch_size, 1).fill_(1.0)
        fake = FloatTensor(opt.batch_size, 1).fill_(0.0)

        # Configure the input
        real_trs = real_trs.type(LongTensor)    # cast real_trs to LongTensor
        labels = labels.type(LongTensor)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise
        z = FloatTensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim)))

        # Sample labels
        gen_labels = LongTensor(np.random.randint(0, opt.n_classes, opt.batch_size))

        # Generate a batch of traces
        gen_trs = generator(z, gen_labels)

        # Measure generator's ability to fool the discriminator
        validity = discriminator(gen_trs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real traces
        real_trs = real_trs.view((50, 1, 2*opt.wnd_size))
        validity_real = discriminator(real_trs, labels)            # prob.trace is real and is compatible with the label
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake traces
        validity_fake = discriminator(gen_trs.detach(), gen_labels)   # prob. trace is fake and is compatible with label
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Update Generator's loss value
        y_g[epoch] = g_loss.item()

        # Update Discriminator's loss value
        y_d[epoch] = d_loss.item()

        d_loss.backward()
        optimizer_D.step()

        '''
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )
        
        batches_done = epoch * len(dataloader) + i
        '''
        
        # Plot the loss
        if sys.platform == 'win32':

            if epoch == opt.n_epochs - 1 and i == opt.batch_size - 1:
                x = np.arange(0, opt.n_epochs)

                plot_gen = plt.figure(1)
                plt.title("Generator's loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.plot(x, y_g)

                plot_disc = plt.figure(2)
                plt.title("Discriminator's loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.plot(x, y_d)

                plt.show()

        print(
            "[Epoch %d/%d]  [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, d_loss.item(), g_loss.item())
        )

np.save('Loss_gen', y_g)
np.save('Loss_disc', y_d)


