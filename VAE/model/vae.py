import torch
import torchvision
import torch.nn as nn
from torch.nn import ModuleDict
import torch.nn.functional as F

from edflow import get_logger
from edflow.custom_logging import LogSingleton
import numpy as np

from model.util import (
    get_tensor_shapes,
    test_config_model,
    complete_config,
    get_act_func,
)
from model.modules import (
    NormConv2d,
    Downsample,
    Upsample
)

class VAE_Model(nn.Module):
    def __init__(self, config):
        super(VAE_Model, self).__init__()
        # set log level to debug if requested
        if "debug_log_level" in config and config["debug_log_level"]:
            LogSingleton.set_log_level("debug")
        # get logger and config
        self.logger = get_logger("VAE_Model")
        # Test the config
        test_config_model(config)
        self.config = complete_config(config, self.logger)
        
        # calculate the tensor shapes throughout the network
        self.tensor_shapes = get_tensor_shapes(config)
        self.logger.info(str(self.tensor_shapes))
        # get the activation function
        self.act_func = get_act_func(config, self.logger)
        self.conv = NormConv2d
        # craete encoder and decoder
        self.enc = VAE_Model_Encoder(config = config, act_func = self.act_func, tensor_shapes = self.tensor_shapes, conv = self.conv, latent_dim = self.config["linear"]["latent_dim"], sigma = bool("variational" in self.config["linear"] and "sigma" in self.config["linear"]["variational"] and self.config["linear"]["variational"]["sigma"]) )
        self.dec = VAE_Model_Decoder(config = config, act_func = self.act_func, tensor_shapes = self.tensor_shapes, conv = self.conv, latent_dim = self.config["linear"]["latent_dim"], sigma = bool("variational" in self.config["linear"] and "sigma" in self.config["linear"]["variational"] and self.config["linear"]["variational"]["sigma"]) )
    
    
    def latent_sample(self, mu, sig = 1):
        """Sample in the latent space. Input a gaussian distribution to retrive an image from the decoder.
        
        :param mu: The expected value of the gaussian distribution. For only one vaule you can specify a float. For interesting sampling should be a Tensor with dimension same as latent dimension.
        :type mu: Tensor or float
        :param sig: The standard deviation of the gaussian distribution. For only one vaule you can specify a float. For interesting sampling should be a Tensor with dimension same as latent dimension if the model is build for a custom standard deviation, defaults to 1
        :type sig: Tensor/float, optional
        :return: Retruns an image according to the sample.
        :rtype: Tensor
        """        
        assert "variational" in self.config["linear"], "If you want to sample from a gaussian distribution to create images you need the key 'variational' in the config."
        if type(mu) in [int,float]:
            mu = torch.ones([self.config["linear"]["latent_dim"]]) * mu
            sig = torch.ones([self.config["linear"]["latent_dim"]]) * sig
        if "sigma" in self.config["linear"]["variational"] and self.config["linear"]["variational"]["sigma"]: 
            x = [mu, sigma]
        else:
            x = mu
        
        x = self.dec(x)
        return x
        
    def forward(self, x):
        """Encodes an image x into the latent represenation z and returns an image generated from that represenation.
        
        :param x: Input Image.
        :type x: Tensor
        :return: Image generated from the latent representation. 
        :rtype: Tensor
        """        
        z = self.enc(x)
        self.logger.debug("encoder output: " + str(z.shape))
        x = self.dec(z)
        self.z = self.dec.z
        self.logger.debug("decoder output: " + str(x.shape))
        #print("decoder output: " + str(x.shape))
        return x 

class VAE_Model_Encoder(nn.Module):
    """This is the encoder for the VAE model."""    
    def __init__(
        self, 
        config, 
        act_func, 
        tensor_shapes,
        conv = NormConv2d,
        latent_dim = None,
        sigma = None
    ):
        super(VAE_Model_Encoder,self).__init__()
        self.logger = get_logger("VAE_Model_Encoder")
        # save all required parameters
        self.config = config 
        self.act_func = act_func
        self.latent_dim = latent_dim
        self.sigma = sigma
        self.tensor_shapes = tensor_shapes
        
        self.setup_modules()
        '''
        # paramers that are needed from config
            # conv
                # all
            # variational but not sigma 
            # upsample
        '''

    def setup_modules(self):
        # Create the convolutional blocks specified in the config.
        conv_modules_list = []
        for i in range(self.config["conv"]["n_blocks"]):
            conv_modules_list.append(
                Downsample(channels = self.config["conv"]["conv_channels"][i], out_channels = self.config["conv"]["conv_channels"][i+1], kernel_size = self.config["conv"]["kernel_size"], stride = self.config["conv"]["stride"], padding = self.config["conv"]["padding"], conv_layer = NormConv2d) 
                )
            conv_modules_list.append(self.act_func)
            if "upsample" in self.config:
                conv_modules_list.append(nn.MaxPool2d(2, stride=2))
        self.conv_seq = nn.Sequential(*conv_modules_list)
        
        # Add if specified a fully connected layer after the convolutions
        if self.latent_dim != None:
            self.lin_layer = nn.Linear(in_features = self.tensor_shapes[self.config["conv"]["n_blocks"] + 1][0], out_features = self.latent_dim)
            # if sigma in the config is true we want to encode a standard deviation too 
            if self.sigma:
                self.lin_sig = nn.Linear(in_features = self.tensor_shapes[self.config["conv"]["n_blocks"] + 1][0], out_features = self.latent_dim)    
        
    def forward(self, x):
        # Apply all modules in the seqence
        x = self.conv_seq(x)
        self.logger.debug("after all conv blocks x.shape: " + str(x.shape))
        if self.latent_dim != None:
            self.logger.debug("Shape is x.shape: " + str(x.shape))
            self.logger.debug("Shape should be: [" + str(self.config["batch_size"]) + "," + str(self.tensor_shapes[self.config["conv"]["n_blocks"] + 1][0]) + "]")
            x = x.view(-1, self.tensor_shapes[self.config["conv"]["n_blocks"] + 1][0])
            self.logger.debug("Shape is x.shape: " + str(x.shape))            
            if self.sigma:
                # maybe absolut value of sigma act ReLu
                x = [self.lin_layer(x), torch.abs(self.lin_sig(x))]
            else:
                x = self.lin_layer(x)
                if "variational" in self.config["linear"]:
                    x = self.act_func(x)    
        return x
    
class VAE_Model_Decoder(nn.Module):
    def __init__(
        self,
        config, 
        act_func, 
        tensor_shapes,
        sigma,
        latent_dim = None,
        conv = NormConv2d
    ):
        super(VAE_Model_Decoder,self).__init__()
        self.logger = get_logger("VAE_Model_Decoder")
        # save all required parameters
        self.config = config
        self.act_func = act_func
        self.conv = conv
        self.latent_dim = latent_dim
        self.sigma = sigma
        self.tensor_shapes = tensor_shapes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.latent_dim != None:
            self.lin_layer = nn.Linear(in_features = self.latent_dim, out_features = self.tensor_shapes[self.config["conv"]["n_blocks"] + 1][0])
        
        self.conv_seq = self.get_blocks()
    
    def add_conv(self, add_kernel_size = 0, padding = 0):
        return nn.ConvTranspose2d(in_channels = self.config["conv"]["conv_channels"][0], out_channels = self.config["conv"]["conv_channels"][0], 
                                        kernel_size = self.config["conv"]["kernel_size"] + add_kernel_size, stride = 1, padding = padding)

    def get_blocks(self):
        upsample_modules_list = []
        for i in range(self.config["conv"]["n_blocks"], 0, -1):
            if "upsample" in self.config:
                if "conv_after_upsample" in self.config and self.config["conv_after_upsample"]:
                    # TODO this has to be fixed somehow
                    print("channels:",self.config["conv"]["conv_channels"][i], "  spacial_size to upsample to:",self.tensor_shapes[i-1][1:])
                    upsample_modules_list.append(self.conv(in_channels = self.config["conv"]["conv_channels"][i], out_channels = self.config["conv"]["conv_channels"][i-1], kernel_size=3, stride=1, padding=1))
                upsample_modules_list.append(nn.Upsample(size = self.tensor_shapes[i-1][1:], mode = self.config["upsample"]))
            else:
                upsample_modules_list.append(Upsample(in_channels = self.config["conv"]["conv_channels"][i], out_channels = self.config["conv"]["conv_channels"][i-1], conv_layer = self.conv))
            if i != 1:
                upsample_modules_list.append(self.act_func)
            else:
                if "final_layer" in self.config["conv"] and self.config["conv"]["final_layer"]:
                    upsample_modules_list.append(self.conv(self.config["conv"]["conv_channels"][0], self.config["conv"]["conv_channels"][0], kernel_size=1))
                upsample_modules_list.append(nn.Tanh())
        return nn.Sequential(*upsample_modules_list)

    def old_get_blocks(self):
        # Get convolutional block with fitting spacial size
        # downsampling with convolutions
        convT_modules_list = []

        for i in range(self.config["conv"]["n_blocks"], 0, -1):
            if "upsample" in self.config and [self.config["conv"]["kernel_size"], self.config["conv"]["stride"]] in [[1,1],[3,1],[5,1]]:
                self.logger.debug("upsample size: " + str(self.tensor_shapes[i-1][1:]))
                convT_modules_list.append(nn.Upsample(size = self.tensor_shapes[i-1][1:], mode = self.config["upsample"]))
            elif "upsample" in self.config and [self.config["conv"]["kernel_size"], self.config["conv"]["stride"]] in [[2,2],[4,4]]:
                spacial_size = int(np.floor(self.tensor_shapes[i-1][-1]/self.config["conv"]["kernel_size"]))
                spacial_size = [spacial_size,spacial_size]
                self.logger.debug("upsample size: " + str(spacial_size))
                convT_modules_list.append(nn.Upsample(size = spacial_size, mode = self.config["upsample"]))
            elif "upsample" in self.config and [self.config["conv"]["kernel_size"], self.config["conv"]["stride"]] in [[3,3]]:
                spacial_size = int(np.floor(self.tensor_shapes[i-1][-1]/self.config["conv"]["kernel_size"]+1))
                spacial_size = [spacial_size,spacial_size]
                self.logger.debug("upsample size: " + str(spacial_size))
                convT_modules_list.append(nn.Upsample(size = spacial_size, mode = self.config["upsample"]))

            convT_modules_list.append(
                    nn.ConvTranspose2d(in_channels = self.config["conv"]["conv_channels"][i], out_channels = self.config["conv"]["conv_channels"][i-1], 
                                        kernel_size = self.config["conv"]["kernel_size"], stride = self.config["conv"]["stride"], 
                                        padding = self.config["conv"]["padding"])
                    )
            if i != 1:
                convT_modules_list.append(self.act_func)
        if not "upsample" in self.config:
            if [self.config["conv"]["kernel_size"], self.config["conv"]["stride"]] in [[3,2]]:
                convT_modules_list.append(self.act_func)
                convT_modules_list.append(self.add_conv(1))
            elif [self.config["conv"]["kernel_size"], self.config["conv"]["stride"]] in [[4,2]]:
                convT_modules_list.append(self.act_func)
                convT_modules_list.append(self.add_conv(1,1))
            elif [self.config["conv"]["kernel_size"], self.config["conv"]["stride"]] in [[5,2]]:
                convT_modules_list.append(self.act_func)
                convT_modules_list.append(self.add_conv(-1))
            elif [self.config["conv"]["kernel_size"], self.config["conv"]["stride"]] in [[4,3]]:
                convT_modules_list.append(self.act_func)
                convT_modules_list.append(self.add_conv(5,1))
            elif [self.config["conv"]["kernel_size"], self.config["conv"]["stride"]] in [[6,3]]:
                convT_modules_list.append(self.act_func)
                convT_modules_list.append(self.add_conv(2))
        
        return nn.Sequential(*convT_modules_list)    

    def forward(self, x):
        if self.latent_dim != None:
            if "variational" in self.config["linear"]:
                norm_dist = torch.distributions.normal.Normal(torch.zeros([self.latent_dim]), torch.ones([self.latent_dim]))
                eps = norm_dist.sample().to(self.device)
                var = 1
                if self.sigma:
                    x = x[0]
                    var = x[1]
                x = x + var * eps
            self.z = x
            x = self.act_func(self.lin_layer(x))
            x = x.reshape(-1,*self.tensor_shapes[self.config["conv"]["n_blocks"]])
        else:
            self.z = x
        print("actual shape:", x.shape)
        x = self.conv_seq(x)
        return x

#TODO log the spacial sizes everywhere in wandb
#TODO get kl loss after a model is trained for a while
#TODO late dev look for alternatives or toggling maxpool2d
#TODO maybe think of a way to better organize 'upsample' in config

#############################
#########  Testing  #########
#############################





'''
import yaml

with open("conv_config.yaml") as fh:
    config_y = yaml.full_load(fh)    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ae = Model(config = config_y)
ae = ae.to(device)
x = torch.zeros([5,3,64,64])
x = x.to(device)
out = ae(x)


enc = encoder(config = config_y, act_func = nn.LeakyReLU(True))
dec = decoder(config_y, nn.ReLU(True))
x = torch.zeros([5,3,64,64])
print("ecoder input:", x.shape)
x = enc(x)
print("encoder output:", x.shape)
print("spacial_res:", dec.get_conv_shape())
x = dec(x)
print("decoder output:", x.shape)
'''