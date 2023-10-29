import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import MultiheadAttention
# correlation matrix normalization + Spatial regularization

class LSTM(nn.Module):
    def __init__(self, time_window, num_point, num_layers=1, output_size=1):
        super(LSTM, self).__init__()
        self.time_window = time_window

        self.lstm = nn.LSTM(time_window, time_window, num_layers, batch_first=True)

        # Define the output layer
        # self.linear = nn.Linear(time_window, 1)
        self.linear = nn.Linear(time_window, 1)

        # Self-attention correlation
        self.values = nn.Linear(time_window, time_window, bias=True)
        self.keys = nn.Linear(time_window, time_window, bias=True)
        self.queries = nn.Linear(time_window, time_window, bias=True)


    def forward(self, x):

        norm_x = F.normalize(x)

        # Initialize LSTM hidden and cell states
        h = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(x.device)
        c = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(x.device)

        z, _ = self.lstm(x, (h,c))

        values = self.values(z)
        keys = self.keys(z)
        queries = self.queries(z)

        # Scaled dot-product attention
        attention = torch.matmul(queries, keys.T) 
        attention = F.softmax(attention / (self.time_window ** 0.5), dim=-1)

        z = torch.matmul(attention, values)

        pred_x = self.linear(z)

        return pred_x