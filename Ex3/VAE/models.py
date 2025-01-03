from imports import *

class Encoder(nn.Module):
    ''' This the encoder part of VAE
    '''

    def __init__(self, hidden_dim, z_dim, type):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            z_dim: A integer indicating the latent dimension.
        '''
        super().__init__()

        self.type = type
        if self.type == 'conv':
            self.l1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
            self.l2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
            self.l3 = nn.Linear(32 * 24 * 24, 512)
            self.l4 = nn.Linear(512, hidden_dim)
        if self.type == 'linear':
            self.l1 = nn.Linear(28 * 28, 600)
            self.l2 = nn.Linear(600, hidden_dim)

        self.mu = nn.Linear(hidden_dim, z_dim)
        self.var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim]
        if self.type == 'conv':
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            bs, ch, w, h = x.shape
            x = x.view(bs, ch * w * h)
            x = F.relu(self.l3(x))
            hidden = F.relu(self.l4(x))
        if self.type == 'linear':
            bs, ch, w, h = x.shape
            x = F.softplus(self.l1(x.view(bs, ch * w * h)))
            hidden = F.softplus(self.l2(x))

        # hidden is of shape [batch_size, hidden_dim]
        z_mu = self.mu(hidden)
        # z_mu is of shape [batch_size, latent_dim]
        z_var = self.var(hidden)
        # z_var is of shape [batch_size, latent_dim]

        return z_mu, z_var


class Decoder(nn.Module):
    ''' This the decoder part of VAE
    '''

    def __init__(self, z_dim, hidden_dim, type):
        super().__init__()

        self.type = type
        if self.type == 'conv':
            self.l1 = nn.Linear(z_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, 512)
            self.l3 = nn.Linear(512, 18432)

            self.l4 = nn.ConvTranspose2d(32, 16, kernel_size=3)
            self.l5 = nn.ConvTranspose2d(16, 1, kernel_size=3)
        if self.type == 'linear':
            self.l1 = nn.Linear(z_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, 28 * 28)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim]
        if self.type == 'conv':
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = F.relu(self.l3(x))
            # print(x.shape)
            x = x.view(-1, 32, 24, 24)
            x = F.relu(self.l4(x))
            predicted = torch.sigmoid(self.l5(x))
        if self.type == 'linear':
            bs = x.shape[0]
            x = F.softplus(self.l1(x))
            predicted = torch.sigmoid(self.l2(x))
            predicted = predicted.view(bs, 1, 28, 28)
        # predicted is of shape [batch_size, output_dim]
        return predicted


class VAE(nn.Module):
    def __init__(self, enc, dec):
        ''' This the VAE, which takes a encoder and decoder.
        '''
        super().__init__()

        self.enc = enc
        self.dec = dec

    def forward(self, x):
        # encode
        z_mu, z_var = self.enc(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        # decode
        predicted = self.dec(x_sample)
        return predicted, z_mu, z_var

def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)