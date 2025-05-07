import torch, torch.nn as nn, torch.nn.functional as F

class CNN2DLSTM(nn.Module):

    def __init__(self, n_ch=64, hidden=128, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (1,7), stride=(1,2), padding=(0,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (n_ch,5), padding=(0,2)),
            nn.ReLU(),
        )
        self.flatten_time = nn.Sequential(
            nn.Conv2d(64, 128, (1,5), stride=(1,2), padding=(0,2)),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=hidden,
            num_layers=1, batch_first=True, bidirectional=False
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.flatten_time(x)
        x = x.squeeze(2).transpose(1,2)
        out, _ = self.lstm(x)
        h_last = out[:,-1]
        return torch.sigmoid(self.head(h_last)).squeeze(1)
