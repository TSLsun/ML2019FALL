#%%
import os
import sys
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.optim import RMSprop, Adam, SGD
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Ref: https://github.com/lucabergamini/VAEGAN-PYTORCH/

train_file = sys.argv[1]
# model_dir = sys.argv[2]
model_dir = "./vae_epoch0350.pth"
pred_file = sys.argv[2]

# use_cuda = torch.cuda.is_available()
# DEVICE = torch.device('cuda' if use_cuda else 'cpu')
DEVICE = torch.device('cpu')
print("Device:", DEVICE)
#%%
# KERNEL_SIZE, PAD = 3, 1
KERNEL_SIZE, PAD = 5, 2

### PARAMS - model, train, test
Z_SIZE = 10
TRAIN_BATCH_SIZE = 30
TEST_BATCH_SIZE = 90
N_EPOCH = 20
LR = 3 * 1e-4
SAVE_FREQ = 50

class RollingMeasure(object):
    def __init__(self):
        self.measure = 0.0
        self.iter = 0

    def __call__(self, measure):
        # passo nuovo valore e ottengo average
        # se first call inizializzo
        if self.iter == 0:
            self.measure = measure
        else:
            self.measure = (1.0 / self.iter * measure) + (1 - 1.0 / self.iter) * self.measure
        self.iter += 1
        return self.measure

class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=KERNEL_SIZE, padding=PAD, stride=2,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)

    def forward(self, x, out=False):
        # here we want to be able to take an intermediate output for reconstruction error
        if out:
            x = self.conv(x)
            tensor_out = x
            x = self.bn(x)
            x = F.relu(x, inplace=False)
            return x, tensor_out
        else:
            x = self.conv(x)
            x = self.bn(x)
            # x = F.relu(x, inplace=True)
            x = F.leaky_relu(x, 0.2, inplace=True)
            return x

class DecoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(DecoderBlock, self).__init__()
        # transpose convolution to double the dimensions
        self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=KERNEL_SIZE, padding=PAD, stride=2, output_padding=1,
                                       bias=False)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = F.relu(x, inplace=True)
        x = F.leaky_relu(x, 0.2, inplace=True)
        return x

class Encoder(nn.Module):
    def __init__(self, channel_in=3, z_size=512):
        super(Encoder, self).__init__()
        self.size = channel_in
        layers_list = []
        # the first time 3->64, for every other double the channel size
        for i in range(3):
            if i == 0:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=64))
                self.size = 64
            else:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=self.size * 2))
                self.size *= 2

        # final shape Bx256x4x4
        self.conv = nn.Sequential(*layers_list)
        self.fc = nn.Sequential(nn.Linear(in_features=4 * 4 * self.size, out_features=1024, bias=False),
                                nn.BatchNorm1d(num_features=1024, momentum=0.9),
                                # nn.ReLU(inplace=True)
                                nn.LeakyReLU(negative_slope=0.2, inplace=True)
                                )
        # two linear to get the mu vector and the diagonal of the log_variance
        self.l_mu = nn.Linear(in_features=1024, out_features=z_size)
        self.l_var = nn.Linear(in_features=1024, out_features=z_size)
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(torch.FloatTensor(std.size()).normal_().to(DEVICE))
        # z = mu + std * esp
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(len(x), -1)
        x = self.fc(x)
        mu = self.l_mu(x)
        logvar = self.l_var(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def __call__(self, *args, **kwargs):
        return super(Encoder, self).__call__(*args, **kwargs)


class Decoder(nn.Module):
    def __init__(self, z_size, size):
        super(Decoder, self).__init__()
        # start from B*z_size
        self.fc = nn.Sequential(nn.Linear(in_features=z_size, out_features=4 * 4 * size, bias=False),
                                nn.BatchNorm1d(num_features=4 * 4 * size, momentum=0.9),
                                # nn.ReLU(inplace=True)
                                nn.LeakyReLU(negative_slope=0.2, inplace=True)
                                )
        self.size = size
        layers_list = []
        # layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size))
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size//2))
        self.size = self.size//2
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size//2))
        self.size = self.size//2
        # layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size//4))
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size//2))
        self.size = self.size//2
        # final conv to get 3 channels and tanh layer
        layers_list.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=3, kernel_size=5, stride=1, padding=2),
            # nn.Tanh()
            nn.Sigmoid()
            # nn.Softmax()
        ))

        self.conv = nn.Sequential(*layers_list)

    def forward(self, x):

        x = self.fc(x)
        x = x.view(len(x), -1, 4, 4)
        x = self.conv(x)
        return x

    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__(*args, **kwargs)

class VAE(nn.Module):
    def __init__(self, z_size=128):
        super(VAE, self).__init__()

        self.z_size = z_size
        self.encoder = Encoder(z_size=self.z_size)
        self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size)
        
        # self-defined function to init the parameters
        self.init_parameters()
    
    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    #init as original implementation
                    scale = 1.0/np.sqrt(np.prod(m.weight.shape[1:]))
                    scale /= np.sqrt(3)
                    #nn.init.xavier_normal_(m.weight,1)
                    #nn.init.constant_(m.weight,0.005)
                    nn.init.uniform_(m.weight,-scale,scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        z, mu, var = self.encoder(x)
        re_x = self.decoder(z)
        
        # return reconstruct & latent
        return re_x, z, mu, var

#%%
def loss_fn(recon_x, x, mu, logvar, mse=False):
    if mse:
        BCE = F.mse_loss(recon_x, x, size_average=False)
    else:
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD
#%%
def denorm(image):
    return (image+1)/2

#%%
# load data and normalize to [-1, 1]
# trainX = np.load('data/trainX.npy')
trainX = np.load(train_file)
trainX = np.transpose(trainX, (0, 3, 1, 2)) / 255.
# trainX = np.transpose(trainX, (0, 3, 1, 2)) / 255. * 2 - 1
trainX = torch.Tensor(trainX)

trainX = trainX.to(DEVICE)

# Dataloader: train shuffle = True
train_dataloader = DataLoader(trainX, batch_size=64, shuffle=True)
test_dataloader = DataLoader(trainX, batch_size=TEST_BATCH_SIZE, shuffle=False)

# model
# model_dir = "VAE_kernel_5_jobs/z20_VCAE_BCE_epoch2000"
model = VAE(z_size=Z_SIZE)  

model.load_state_dict(torch.load(model_dir, map_location=DEVICE))
model = model.to(DEVICE)
#%%
# Collect the latents and stdardize it.
model.eval()

latents = []
reconstructs = []
for x in test_dataloader:
    reconstruct, latent, _, _ = model(x)
    latents.append(latent.cpu().detach().numpy())
    reconstructs.append(reconstruct.cpu().detach().numpy())

latents = np.concatenate(latents, axis=0).reshape([9000, -1])
latents = (latents - np.mean(latents, axis=0)) / np.std(latents, axis=0)
    

#%%
# Use PCA to lower dim of latents and use K-means to clustering.
# latents = PCA(n_components=32).fit_transform(latents)
print("TSNE processing...", flush=True)
latents = TSNE(n_components=2, random_state=777).fit_transform(latents)
print("Clustering...", flush=True)
result = KMeans(n_clusters=2, random_state=666, n_jobs=-1).fit(latents).labels_
# result = SpectralClustering(n_clusters=2, random_state=666, n_jobs=-1).fit(latents).labels_

# We know first 5 labels are zeros, it's a mechanism to check are your answers
# need to be flipped or not.
if np.sum(result[:5]) >= 3:
    result = 1 - result

# Generate your submission
df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
df.to_csv(pred_file, index=False)