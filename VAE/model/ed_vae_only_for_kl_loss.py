import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler


import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader



import numpy as np
import matplotlib.pyplot as plt
from edflow import TemplateIterator, get_logger
from edflow.data.dataset import DatasetMixin

import os
from skimage import io, transform
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.enc = encoder(config["latent_dim"]).to(device)
        self.dec = decoder(config["latent_dim"]).to(device)
    
    def forward(self, x):
        if type(x)==dict:
            x = x["image"]
        x=x.to(device)
        mu, var = self.enc(x)
        
        norm_dist = torch.distributions.normal.Normal(0, 1)
        eps = norm_dist.sample()
        z = mu + var * eps
        pred = self.dec(z)
        
        return pred, mu, var
    
    def variation(self, mu, var):
        norm_dist = torch.distributions.normal.Normal(0, 1)
        eps = norm_dist.sample()
        z = mu + var * eps
        return self.dec(z)


class encoder(nn.Module):
    def __init__(self, latent_dim):
        super(encoder,self).__init__()
        self.lin_mu = nn.Linear(18,latent_dim)
        self.lin_var = nn.Linear(18,latent_dim)
        
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels= 9, kernel_size= 5, stride=3, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(9, 18, 5, stride=3, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=3)
        )

    def forward(self,x):
        if type(x)==dict:
            x = x["image"]
        x=x.to(device)
        x = self.seq(x).view(-1,18)
        return self.lin_mu(x), torch.sqrt(self.lin_var(x)**2)
        
class decoder(nn.Module):
    def __init__(self, latent_dim):
        super(decoder,self).__init__()
        self.lin = nn.Linear(latent_dim,36)
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(36, 18, 5, stride=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(18, 12, 5, stride=3, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 9, 5, stride=3, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(9, 3, 4, stride=2, padding=0),
            nn.Tanh()
            )
      
    def forward(self, x):
        if type(x)==dict:
            x = x["image"]
        x=x.to(device)
        x = self.lin(x).unsqueeze(0).unsqueeze(0)
        x = x.permute(2,3,0,1)
        return self.seq(x).to(device)


    

'''
learning_rate = 0.0001
latent_dim = 10

model_vae = VAE_Net(latent_dim=latent_dim).to(device)
#model_two_cuboids.load_state_dict(torch.load("data/model/newest.pth"))
model_vae.load_state_dict(torch.load("/export/home/rhaecker/documents/VAE/data/vae_max_var/model/NN_state_vae_new_eq_latent_1_150.pth"
                                    ))

data = CuboidDataset("/export/home/rhaecker/documents/VAE/data/vae_max_var",batch_size=5)


wandb.config.learning_rate = 0.0001
#wandb.config.bottleneck = 5
wandb.config.update({"config.bottleneck":10,"WANDB_NOTEBOOK_NAME":"vae_max_var"},allow_val_change=True)
wandb.config.criterion = "MSE_Loss"
wandb.config.weight_decay = 1e-5
wandb.config.optimizer = "Adam"
#wandb.config.WANDB_NOTEBOOK_NAME = "vae_only_phi_new"
'''