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

#TODO Wandb implemantation solve problem and plotting and saving images and change to dev branch

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
        return self.seq(x)



class Dataset(DatasetMixin):
    def __init__(self, config, train=False):
        """Initialize the dataset to load training or validation images according to the config.yaml file. 
        
        :param DatasetMixin: This class inherits from this class to enable a good workflow through the framework edflow.  
        :param config: This config is loaded from the config.yaml file which specifies all neccesary hyperparameter for to desired operation which will be executed by the edflow framework.
        """
        # Create Logging for the Dataset
        self.logger = get_logger("Dataset")
        
        # Get the directory to the data and format it
        assert "data_path" in config, "You have to specify the directory to the data in the config.yaml file."
        self.data_path = config["data_path"]
        if "~" in self.data_path:
            self.data_path = os.path.expanduser('~') + self.data_path[self.data_path.find("~")+1:]
        self.logger.debug("data_path: " + str(self.data_path))
        
        # Transforming and resizing images
        if "image_resolution" in config:    
            if type(config["image_resolution"])!=list:
                config["image_resolution"]=[config["image_resolution"], config["image_resolution"]]
            self.transform = torchvision.transforms.Compose([transforms.Resize(size=(config["image_resolution"][0],config["image_resolution"][1])), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
            self.logger.debug("Resizing images to " + str(config["image_resolution"]))
        else:
            self.logger.info("Images will not be resized! Original image resolution will be used.")
            self.transform = torchvision.transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        
        # Load parameters from config
        self.batch_size = config["batch_size"]
        self.latent_dim = config["latent_dim"]
        # Load every indices from all images
        all_indices = [int(s[12:-4]) for s in os.listdir(self.data_path + "/images/")]
        
        # Split data into validation and training data 
        split = int(np.floor(config["validation_split"] * len(all_indices)))
        if config["shuffle_dataset"]:
            if "random_seed" in config:
                np.random.seed(config["random_seed"])
            np.random.shuffle(all_indices)        
        # Load training or validation images as well as their indices
        if train:
            self.indices = all_indices[split:]
            train_sampler = SubsetRandomSampler(self.indices)
            self.dataset = torch.utils.data.DataLoader(self, batch_size=self.batch_size, sampler=train_sampler)
        else:
            self.indices = all_indices[:split]
            valid_sampler = SubsetRandomSampler(self.indices)
            self.dataset = torch.utils.data.DataLoader(self, batch_size=self.batch_size, sampler=valid_sampler)
        self.actual_epoch = 0

    def __len__(self):
        """This member function returns the length of the dataset
        
        :return: [description]
        :rtype: [type]
        """
        return len(self.indices)

    def get_example(self, idx):
        """This member function loads and returns images in a dictionary according to the given index.
        
        :param idx: Index of the requested image.
        :type idx: Int
        :return: Dictionary with the image at the key 'image'.
        :rtype: Dictionary
        """
        idx = self.indices[int(idx)]
        # Load image
        img_name = os.path.join(self.data_path, "images/image_index_" + str(idx) + ".png")
        image = io.imread(img_name)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        # Return dictionary
        sample = {'image': image}
        return sample

    def plot(self, image, name=None):
        if type(image)==dict:
            image = image["image"]
        with torch.no_grad():
            if image.shape[0]==self.batch_size:
                te = torch.zeros(self.batch_size,1024,1024,3,requires_grad=False)
                te = torch.Tensor.permute(image,0,2,3,1)
                plt.figure(figsize=(20,5))
                te = (te/2+0.5).cpu()
                te.detach().numpy()
                plot_image = np.hstack(te.detach())
            else:
                te = torch.zeros(3,1024,1024,requires_grad=False)
                te = torch.Tensor.permute(image,1,2,0)
                te = (te/2+0.5).cpu()
                te.detach().numpy()
                plot_image = te
            plt.imshow(plot_image)
            if name!=None:
                path_fig=self.data_path + "/figures/first_run_150e/"
                if not os.path.isdir(path_fig):
                    os.mkdir(path_fig)
                plt.savefig( path_fig + "figure_latent_dim_"+ str(self.latent_dim) +"_"+str(name)+".png")
            plt.show()

class DatasetTrain(Dataset):
    def __init__(self, config):
        super().__init__(config, train=True)
    
class DatasetEval(Dataset):
    def __init__(self, config):
        super().__init__(config, train=False)
    

class Iterator(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # loss and optimizer
        self.logger = get_logger("Iterator")
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-5)

    def save(self, checkpoint_path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

    def step_op(self, model, **kwargs):
        # get inputs
        inputs = kwargs["image"]
        inputs = torch.tensor(inputs)
        #inputs = inputs.permute(0, 3, 1, 2)
        
        #labels = torch.tensor(inputs)
        #labels = labels.permute(0, 3, 1, 2)
        #labels = torch.tensor(labels, dtype=torch.long)

        # compute loss
        outputs, mu , var = model(inputs)
        self.logger.debug("inputs.shape" + str(inputs.shape))
        self.logger.debug("output.shape:" + str(outputs.shape))
        
        loss = self.criterion(outputs, inputs)# check shape and stuff
        mean_loss = torch.mean(loss)

        def train_op():
            self.optimizer.zero_grad()
            mean_loss.backward()
            self.optimizer.step()

        def log_op():
            '''acc = np.mean(
                np.argmax(outputs.detach().numpy(), axis=1) == labels.detach().numpy()
            )
            '''
            min_loss = np.min(loss.detach().numpy())
            max_loss = np.max(loss.detach().numpy())
            return {
                "images": {"inputs": inputs.detach().permute(0, 2, 3, 1).numpy()},#,"outputs": outputs.detach().permute(0, 2, 3, 1).numpy()},
                "scalars": {
                    "min_loss": min_loss,
                    "max_loss": max_loss,
                    "mean_loss": mean_loss,
                    #"acc": acc,
                },
            }

        def eval_op():
            return {
                "outputs": np.array(outputs.detach().permute(0,2,3,1).numpy()),
                "labels": {"loss": np.array(loss.detach().numpy())},
            }

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

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