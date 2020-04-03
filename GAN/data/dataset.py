import torch
import torchvision
import torchvision.transforms as transforms

from edflow import get_logger
from edflow.data.dataset import DatasetMixin
from edflow.util import edprint

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
from PIL import Image
import json

class Dataset(DatasetMixin):
    def __init__(self, config, train=False):
        """Initialize the dataset to load training or validation images according to the config.yaml file. 
        
        :param DatasetMixin: This class inherits from this class to enable a good workflow through the framework edflow.  
        :param config: This config is loaded from the config.yaml file which specifies all neccesary hyperparameter for to desired operation which will be executed by the edflow framework.
        """
        # Create Logging for the Dataset
        self.logger = get_logger("Dataset")
        
        # Get the directory to the data and format it
        assert "data_root" in config, "You have to specify the directory to the data in the config.yaml file."
        self.data_root = config["data_root"]
        if "~" in self.data_root:
            self.data_root = os.path.expanduser('~') + self.data_root[self.data_root.find("~")+1:]
        self.logger.debug("data_root: " + str(self.data_root))
        
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
        self.config = config
        self.set_random_state()
        # Load every indices from all images
        if "request_tri" in self.config and self.config["request_tri"]:
            every_indices = [int(s[12:-6]) for s in os.listdir(self.data_root + "/images/")]
            all_indices = []
            for i in range(int(np.floor(len(every_indices)/3))):
                all_indices.append(every_indices[i*3])
        else:
            all_indices = [int(s[12:-4]) for s in os.listdir(self.data_root + "/images/")]
        
        # Split data into validation and training data 
        split = int(np.floor(config["validation_split"] * len(all_indices)))
        if self.config["shuffle_dataset"]:
            np.random.shuffle(all_indices)        
        # Load training or validation images as well as their indices
        if train:
            self.indices = all_indices[split:]
        else:
            self.indices = all_indices[:split]

    def set_random_state(self):
        if "random_seed" in self.config:
            np.random.seed(self.config["random_seed"])
            torch.random.manual_seed(self.config["random_seed"])
        else:
            self.config["random_seed"] = np.random.randint(0,2**32-1)
            np.random.seed(self.config["random_seed"])
            torch.random.manual_seed(self.config["random_seed"])

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
        
        '''# load a json file with all parameters which define the image 
        if "request_parameters" in self.config and self.config["request_parameters"]:
            parameter_path = os.path.join(self.data_root, "parameters/parameters_index_" + str(idx) + ".json")
            with open(parameter_path) as f:
              parameters = json.load(f)
            example["parameters"] = parameters
        '''
        example = {}
        idx = self.indices[int(idx)]
        if "request_tri" in self.config and self.config["request_tri"]:
            images = []
            for i in range(3):
                img = (self.load_image(idx = str(idx) + "_" + str(i)))
                img = np.transpose(img, (1, 2, 0))
                images.append(img)
            example["appearance"] = images[0]
            example["stickman"] = images[1]
            example["target"] = images[2]
        else:
            if "request_parameters" in self.config and self.config["request_parameters"]:
                parameters = self.load_parameters(idx)
                example["parameters"] = parameters
            # Load image
            image = self.load_image(idx)
            example["image"] = image
        # Return example dictionary
        return example

    def load_parameters(self, idx):
        # load a json file with all parameters which define the image 
        parameter_path = os.path.join(self.data_root, "parameters/parameters_index_" + str(idx) + ".json")
        with open(parameter_path) as f:
            parameters = json.load(f)
        return parameters
            
    def load_image(self, idx):
        image_path = os.path.join(self.data_root, "images/image_index_" + str(idx) + ".png")
        image = Image.fromarray(io.imread(image_path))
        if self.transform:
            image = self.transform(image)
        return image

    def plot(self, image, name=None):
        if type(image)==dict:
            image = image["image"]
        with torch.no_grad():
            if image.shape[0]==self.config["batch_size"]:
                te = torch.zeros(self.config["batch_size"],1024,1024,3,requires_grad=False)
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
                path_fig=self.data_root + "/figures/first_run_150e/"
                if not os.path.isdir(path_fig):
                    os.mkdir(path_fig)
                #plt.savefig( path_fig + "figure_latent_dim_"+ str(self.latent_dim) +"_"+str(name)+".png")
            plt.show()

class DatasetTrain(Dataset):
    def __init__(self, config):
        super().__init__(config, train=True)
    '''self.P = Dataset(config)
        self.data = self.P.dataset
    '''
class DatasetEval(Dataset):
    def __init__(self, config):
        super().__init__(config, train=False)
        #self.data = self.dataset
#TODO save the random seed somewhere if not specified