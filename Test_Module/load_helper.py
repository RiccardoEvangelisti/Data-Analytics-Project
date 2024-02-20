from torch.utils.data import Dataset
import torch
from torch import nn


# Dataset for FF-NN model
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]



class NeuralNetworkDropoutBatchNorm(nn.Module):
    def __init__(self, input_size, hidden_size, depth, dropout_rate):
        super(NeuralNetworkDropoutBatchNorm, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.hidden_layers = nn.Sequential(
            *[
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
            * (depth - 1)
        )

        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_in = self.input_layer(x)
        h_out = self.hidden_layers(h_in)
        out = self.output(h_out)
        return out

    def _get_description(self):
        return "[Linear(hidden_size)+BatchNorm1d+ReLU+Dropout]*depth"
