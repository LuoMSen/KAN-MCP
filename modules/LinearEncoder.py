import torch.nn as nn
from torch.nn import MSELoss


class LinearEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(LinearEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.loss_fct = MSELoss()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.fc = nn.Linear(latent_dim, 1)

    def forward(self, x, label_ids):
        encoder = self.encoder(x)
        out = self.fc(encoder)
        loss = self.loss_fct(out, label_ids)
        return encoder, out, loss

