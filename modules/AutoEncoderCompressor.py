import torch.nn as nn
from torch.nn import MSELoss


class AutoEncoderCompressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(AutoEncoderCompressor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.loss_fct = MSELoss()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        loss = self.loss_fct(decoder, x)
        return encoder, decoder, loss

