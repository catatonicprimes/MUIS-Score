import torch.nn as nn

class MUISNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate, activation_fn):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if activation_fn == 'relu':
                layers.append(nn.ReLU())
            elif activation_fn == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation_fn == 'elu':
                layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze(-1)
