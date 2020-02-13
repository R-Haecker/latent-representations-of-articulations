import numpy as np
import matplotlib.pyplot as plt
from edflow import TemplateIterator, get_logger
from edflow.data.dataset import DatasetMixin

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import matplotlib.pyplot as plt

from edflow import TemplateIterator, get_logger
from edflow.data.dataset import DatasetMixin

import os
from skimage import io, transform
from PIL import Image


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
        self.latent_dim = config["bottleneck_size"]
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

#TODO save the random seed somewhere if not specified