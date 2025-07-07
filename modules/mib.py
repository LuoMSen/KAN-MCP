import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss

beta = 1e-3

class mib(nn.Module):
    def __init__(self, in_dim, compressed_dim, m_dim=1024):
        super().__init__()

        self.in_dim = in_dim
        self.compressed_dim = compressed_dim
        self.m_dim = m_dim

        # build encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.m_dim),
            nn.ReLU(inplace=True)
        )

        self.fc_mu = nn.Linear(self.m_dim, self.compressed_dim)
        self.fc_std = nn.Linear(self.m_dim, self.compressed_dim)

        # build decoder
        self.decoder = nn.Linear(self.compressed_dim, 1)

    def encode(self, x):
        """
        x : [batch_size,784]
        """
        x = self.encoder(x)
        return self.fc_mu(x), F.softplus(self.fc_std(x)-5, beta=1)

    def decode(self, z):

        return self.decoder(z)

    def reparameterise(self, mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]
        """
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std*eps

    def loss_function(self, y_pred, y, mu, std):

        loss_fct = L1Loss()

        CE = loss_fct(y_pred.view(-1,), y.view(-1,))
        KL = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
        return (beta*KL + CE)


    def forward(
            self,
            x,
            label_ids=None
    ):

        mu, std = self.encode(x)
        # x_l -> z_l
        z = self.reparameterise(mu, std) 
        output = self.decode(z)
        if label_ids is not None:
            loss = self.loss_function(output, label_ids, mu, std)
        else:
            loss = 0

        return z, output, loss


