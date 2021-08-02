import torch
import torch.nn as nn


class LinearNorm(nn.Module):
    def __init__(self, lstm_hidden, emb_dim):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(lstm_hidden, emb_dim)

    def forward(self, x):
        return self.linear_layer(x)


class SpeechEmbedder(nn.Module):
    def __init__(self, num_mels=40, emb_dim=256, lstm_hidden=768, lstm_layers=3, window=80, stride=40):
        super(SpeechEmbedder, self).__init__()
        self.lstm = nn.LSTM(num_mels,
                            lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True)
        self.proj = LinearNorm(lstm_hidden, emb_dim)
        self.window = window
        self.stride = stride 

    def forward(self, mel):
        # (num_mels, T)
        mels = mel.unfold(1, self.window, self.stride) # (num_mels, T', window)
        mels = mels.permute(1, 2, 0) # (T', window, num_mels)
        x, _ = self.lstm(mels) # (T', window, lstm_hidden)
        x = x[:, -1, :] # (T', lstm_hidden), use last frame only
        x = self.proj(x) # (T', emb_dim)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True) # (T', emb_dim)
        x = x.sum(0) / x.size(0) # (emb_dim), average pooling over time frames
        return x
