import argparse
import os
import numpy as np
import pickle

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
parser.add_argument("--trace_samples", type=int, default=80000, help="number of samples")
parser.add_argument("--n_data", type=int, default=100000, help="number of traces")

opt = parser.parse_args()
print(opt)

trace_shape = (1,opt.trace_samples)

cuda = True if torch.cuda.is_available() else False

# ----------------
# Load the dataset
# ----------------

path = '/Users/magdalenasolitro/Desktop/AI&CS MSc. UniUD/Small Project in CS/first order masked AES-128 2 rounds 2/'

traces = None

# every element in the directory 'traces' is appended to an array
for file in os.listdir(path + 'traces'):
    with open(path + 'traces/' + file, 'rb') as f:
        if 'random_traces' in file and not 'test' in file:
            if traces is None:
                traces = np.array(pickle.load(f))
            else:
                trace_temp = np.array(pickle.load(f))
                traces = np.append(traces,trace_temp,axis=0)

# load the labels
labels = np.load(path + 'labels/s1.npy')

# ------------------------
# Create the training set
# ------------------------

# Create list of indexes
idx = np.arange(opt.n_data)

# Associate IDs to labels
labels_dict = {idx[0] : labels[0]}

for i in range(opt.n_data):
    new_dict = {idx[i] : labels[i]}
    labels_dict.update(new_dict)


# Create the training set
training_set = TraceDataset(idx, labels_dict, traces)

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

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        # One block = one Linear unit, followed by a LeakyReLU activation function
        def block(in_feat, out_feat, normalize=True):
            # Linear transformation to the incoming data
            layers = [nn.Linear(in_feat, out_feat)]         # in_feat = size of each input sample
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
            nn.Linear(1024, int(np.prod(trace_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and noise to produce input of the generator
        gen_input = torch.cat((self.label_emb(labels), noise), -1)

        # Generate a trace out of gen_input, passing it through the model
        trace = self.model(gen_input)

        # Reshape the tensor according to trace_shape (1st arg)
        # trace = trace.view(opt.trace_samples)

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
        print("Label embedding: " + str(self.label_embedding(labels).size()))
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
    for i,(real_trs, labels) in enumerate(dataloader):

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

        # Sample label
        # gen_labels = array containing 16 random values. Each value is comprised
        # between 0 and 255 (1 byte = 8 bits = 256 possible values)
        gen_labels = LongTensor(np.random.randint(0, opt.n_classes, (1,16)))

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
        # EXCEPTION
        validity_real = discriminator(real_trs, labels)             # prob. that the trace is real and is compatible with the label
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake traces
        validity_fake = discriminator(gen_trs.detach(), gen_labels) # prob. that the trace is fake and is compatible with the label
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

