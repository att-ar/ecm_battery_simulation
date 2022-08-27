import torch
from torch import nn
from torch.nn.modules.activation import Sigmoid
import numpy as np
import pandas as pd

class LSTMNetwork(nn.Module):
    def __init__(self):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(3, 256, 1, batch_first = True)
        self.linear_stack = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256, momentum = 0.92),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256, momentum = 0.92),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            Sigmoid()
        )
    def forward(self, x):
        #lstm
        x_out, (h_n_lstm, c_n)  = self.lstm(x)
        out = self.linear_stack(h_n_lstm.squeeze())
        return out


def train_loop(dataloader, model, loss_fn, optimizer, progress):
    model.train()
    for batch, (x,y) in enumerate(dataloader):
        progress.progress(batch/(len(dataloader)*2))
        optimizer.zero_grad()
        predict = model(x)
        loss = loss_fn(predict, y).mean(0) # assert(loss.shape == (1))
        loss.backward()
        optimizer.step()

def log_cosh_loss(y_pred, y_ground):
    x = y_pred - y_ground
    return x + torch.nn.functional.softplus(-2. * x) - np.log(2.0)

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                y_pred: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)
