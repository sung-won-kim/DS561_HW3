import torch.nn as nn
import torch
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, time_window, num_point, num_layers=1, output_size=1):
        super(LSTM, self).__init__()
        self.time_window = time_window

        self.lstm = nn.LSTM(time_window, time_window, num_layers, batch_first=True)

        # Define the output layer
        # self.linear = nn.Linear(time_window, 1)
        self.linear = nn.Linear(time_window+num_point, 1)

    def forward(self, x):

        # Initialize LSTM hidden and cell states
        h = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(x.device)
        c = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(x.device)

        # correlation_matrix = F.normalize(torch.matmul(x, x.T))
        # I = torch.matmul(correlation_matrix, x)
        z, _ = self.lstm(x, (h,c))
        correlation_matrix = F.normalize(torch.matmul(z, z.T))

        z = torch.concat((z, correlation_matrix),1)
        pred_x = self.linear(z)

        return pred_x