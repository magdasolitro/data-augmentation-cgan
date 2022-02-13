import torch

class TraceDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, traces):
        'Initialization'
        self.list_IDs = list_IDs
        self.labels = labels
        self.traces = traces

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = self.traces[index]
        y = self.labels[index][0]   # only first byte

        return X, y
