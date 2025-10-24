import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, hidden_dim: int):
        super(SimpleLSTM, self).__init__()
        input_dim = 1   # single input feature
        output_dim = 1  # single output feature
        self.hidden_dim = hidden_dim

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Map hidden state to output
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, input_dim)
        # hidden: optional tuple (h_0, c_0)
        out, (h_n, c_n) = self.lstm(x, hidden)
        y = self.fc(out)  # shape: (batch_size, seq_len, output_dim)
        return y, (h_n, c_n)