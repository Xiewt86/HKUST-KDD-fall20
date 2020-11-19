from torch import nn
import torch


class SiameseNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm_seek = Subnet(input_size)
        self.lstm_serve = Subnet(input_size)
        self.fc_profile = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        self.fc_sim = nn.Sequential(
            nn.Linear((48*2+8)*2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x_seek1, x_serve1, x_seek2, x_serve2, x_profile1, x_profile2):
        embed_seek1 = self.lstm_seek(x_seek1)
        embed_serve1 = self.lstm_serve(x_serve1)
        embed_profile1 = self.fc_profile(x_profile1)

        embed_seek2 = self.lstm_seek(x_seek2)
        embed_serve2 = self.lstm_serve(x_serve2)
        embed_profile2 = self.fc_profile(x_profile2)

        embed1 = torch.cat((embed_seek1, embed_serve1, embed_profile1), 1)
        embed2 = torch.cat((embed_seek2, embed_serve2, embed_profile2), 1)

        embed = torch.cat((embed1, embed2), 1)

        out = self.fc_sim(embed)
        return out


class Subnet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 200, batch_first=True)
        self.lstm2 = nn.LSTM(200, 100, batch_first=True)
        self.fc1 = nn.Linear(100, 48)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        embedding = self.fc1(out[:, -1, :])
        return embedding
