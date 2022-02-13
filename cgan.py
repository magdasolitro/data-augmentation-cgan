import argparse
import os
import numpy as np

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
parser.add_argument("--n_classes", type=int, default=256, help="number of classes for dataset")          # number of classes = number of possible subkeys
parser.add_argument("--trace_samples", type=int, default=51250, help="number of samples")
parser.add_argument("--n_data", type=int, default=10000, help="number of traces")
parser.add_argument("--target_op", type=str, default='s', help="the intermediate result we want to attack")
parser.add_argument("--wnd_size", type=int, default=250, help="window size around a time point")

opt = parser.parse_args()
print(opt)

trace_shape = (1, 2 * opt.wnd_size)

cuda = True if torch.cuda.is_available() else False


# ----------------
# Load the dataset
# ----------------

path = '/Users/magdalenasolitro/Desktop/AI&CS MSc. UniUD/Small Project in CS/dataset_joey/'

# retrieve significant trace window around the first timepoint
time_points = np.load(path + 'timepoints/' + opt.target_op + '.npy')
this_timepoint = time_points[0]
start = this_timepoint - opt.wnd_size
end = this_timepoint + opt.wnd_size

# WARNING! temporarily working with just one file, loading all of them is too much :c
traces = np.load(path + 'tracedata/random_keys_traces_0.npy', allow_pickle=True)

# keep only the trace portion in the significant window
trimmed_traces = np.array([], dtype=int)

for r in range(traces.shape[0]):
    tmp = traces[r, :]      # select r-th row
    tmp = tmp[start:end]   # select samples in the significant window
    trimmed_traces = np.append(trimmed_traces, tmp, axis=0)

n_columns = end-start
trimmed_traces = np.reshape(trimmed_traces, (traces.shape[0], n_columns))       # new shape: (10000, 3343)

# load the labels
labels = np.load(path + 'realvalues/' + opt.target_op + '.npy')

# Create list of indexes
idx = np.arange(opt.n_data)

# Associate IDs to labels
labels_dict = {idx[0]: labels[:, 0]}

for i in range(opt.n_data):
    new_dict = {idx[i]: labels[:, i]}
    labels_dict.update(new_dict)


# Create the training set
training_set = TraceDataset(idx, labels_dict, trimmed_traces)

# Set dataloader parameters
dataloader = torch.utils.data.DataLoader(
    training_set,
    batch_size=opt.batch_size,
    shuffle=True,
)


# ----------
# Generator
# ----------

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 1 arg =  size of the dictionary of embeddings, 2 arg = the size of each embedding vector
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            # dim. input = gen_labels length (after embedding) + noise vector length
            nn.Linear(opt.n_classes + opt.latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2 * opt.wnd_size),
            nn.Tanh()
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

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            # dim. input = dim. output generator + label length
            nn.Linear(2 * opt.wnd_size + opt.n_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),      # output dim = 1 = prob. that the trace is real
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

# ----------
#  Training
# ----------

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

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_trs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real traces
        validity_real = discriminator(real_trs, labels)            # prob.trace is real and is compatible with the label

        d_real_loss = adversarial_loss(validity_real, valid)
        # Loss for fake traces
        validity_fake = discriminator(gen_trs.detach(), gen_labels)   # prob. trace is fake and is compatible with label
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

