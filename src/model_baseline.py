import torch
import torch.nn as nn

class MLPLayer(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_input, n_output))
        nn.init.xavier_uniform_(self.W)
        self.b = nn.Parameter(torch.zeros(n_output))

    def forward(self, x):
        # x:(B, n_input)
        # out: (B, n_output)
        return x @ self.W + self.b

class MLP(nn.Module): 
    def __init__(self, vocab_size, emb_dim, n_input, n_hidden, n_output):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.layer1 = MLPLayer(n_input, n_hidden)
        self.layer2 = MLPLayer(n_hidden, n_hidden)
        self.layer3 = MLPLayer(n_hidden, n_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, T_in)
        x = self.embedding(x)                  # (B, T_in, emb_dim)
        x = x.reshape(x.size(0), -1)           # (B, T_in * emb_dim)

        y = self.layer1(x)      # 16,512
        y = self.relu(y)
        y = self.layer2(y)      
        y = self.relu(y)
        y = self.layer3(y)
        return y