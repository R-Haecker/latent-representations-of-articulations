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
    complete_config,
    get_act_func,
    test_config,
    set_random_state
)
from model.modules import (
    NormConv2d,
    Downsample,
    Upsample,
    One_sided_padding
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
        test_config(config)
        self.config = complete_config(config, self.logger)
        set_random_state(self.config)
        # calculate the tensor shapes throughout the network
        self.tensor_shapes = get_tensor_shapes(config)
        self.tensor_shapes_dec = get_tensor_shapes(config, encoder = False)
        self.logger.info(str(self.tensor_shapes))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # extract information from config
        self.linear      = bool("linear" in self.config)
        self.variational = bool("variational" in self.config) 
        self.sigma       = bool(self.variational and "sigma" in self.config["variational"] and self.config["variational"]["sigma"])
        if self.linear or self.variational:
            if self.sigma:
                self.latent_dim = int(self.tensor_shapes[-1][-1]/2)
                self.logger.debug("decoder shapes: " + str(self.tensor_shapes_dec))
            else:
                self.latent_dim = self.tensor_shapes[-1][-1]
        else:
            self.latent_dim = None
        # get the activation function
        self.act_func = get_act_func(config, self.logger)
        # craete encoder and decoder
        self.enc = VAE_Model_Encoder(config = config, act_func = self.act_func, tensor_shapes = self.tensor_shapes,     linear = self.linear, variaional = self.variational, sigma = self.sigma, latent_dim = self.latent_dim)
        self.dec = VAE_Model_Decoder(config = config, act_func = self.act_func, tensor_shapes = self.tensor_shapes_dec, linear = self.linear, variaional = self.variational, sigma = self.sigma, latent_dim = self.latent_dim)
    
    def direct_z_sample(self, z):
        z = z.to(self.device)
        x = self.dec(z)
        return x

    def latent_sample(self, mu, var = 1, batch_size = None):
        """Sample in the latent space. Input a gaussian distribution to retrive an image from the decoder.
        
        :param mu: The expected value of the gaussian distribution. For only one vaule you can specify a float. For interesting sampling should be a Tensor with dimension same as latent dimension.
        :type mu: Tensor or float
        :param sig: The standard deviation of the gaussian distribution. For only one vaule you can specify a float. For interesting sampling should be a Tensor with dimension same as latent dimension if the model is build for a custom standard deviation, defaults to 1
        :type sig: Tensor/float, optional
        :return: Retruns an image according to the sample.
        :rtype: Tensor
        """        
        assert "variational" in self.config, "If you want to sample from a gaussian distribution to create images you need the key 'variational' in the config."
        if batch_size == None:
            batch_size = self.config["batch_size"]
        if type(mu) in [int,float]:
            mu  = torch.ones([batch_size, self.latent_dim]).to(self.device) * mu
            var = torch.ones([batch_size, self.latent_dim]).to(self.device) * var
        else:
            assert mu.shape[-1] == self.latent_dim, "Wrong shape for latent vector mu"
        if "sigma" not in self.config["variational"] or not self.config["variational"]["sigma"]:
            if var != 1:
                self.logger.info("Variational: sigma is false, var will be overwritten and set to one")
                var = 1
        norm_dist = torch.distributions.normal.Normal(torch.zeros([batch_size, self.latent_dim]), torch.ones([batch_size, self.latent_dim]))
        eps = norm_dist.sample()
        eps = eps.to(self.device)
        z = mu + var * eps
        z = z.to(self.device)
        x = self.dec(z)
        return x

    def bottleneck(self, x):
        if self.variational:
            norm_dist = torch.distributions.normal.Normal(torch.zeros([x.shape[0], self.latent_dim]), torch.ones([x.shape[0], self.latent_dim]))
            eps = norm_dist.sample().to(self.device)
            if self.sigma:
                self.mu  = x[:, :self.latent_dim]
                self.var = torch.abs(x[:, self.latent_dim:]) + 0.0001
                self.logger.debug("varitaional mu.shape: " + str(self.mu.shape))
                self.logger.debug("varitaional var.shape: " + str(self.var.shape))
            else:
                self.mu  = x
                self.var = 1
                self.logger.debug("varitaional mu.shape: " + str(self.mu.shape))
            # final latent representatione
            x = self.mu + self.var * eps
        return x
        
    def encode_images_to_z(self, x):
        x = self.enc(x)
        self.z = self.bottleneck(x)
        self.logger.debug("output: " + str(self.z.shape))
        return self.z
        

    def forward(self, x):
        """Encodes an image x into the latent represenation z and returns an image generated from that represenation.
        
        :param x: Input Image.
        :type x: Tensor
        :return: Image generated from the latent representation. 
        :rtype: Tensor
        """        
        x = self.enc(x)
        self.z = self.bottleneck(x)
        self.logger.debug("output: " + str(self.z.shape))
            
        x = self.dec(self.z)
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
        conv       = NormConv2d,
        linear     = None, 
        variaional = None, 
        sigma      = None, 
        latent_dim = None
    ):
        super(VAE_Model_Encoder,self).__init__()
        self.logger = get_logger("VAE_Model_Encoder")
        # save all required parameters
        self.config = config 
        self.act_func = act_func
        self.tensor_shapes = tensor_shapes
        self.conv       = conv
        self.linear     = linear 
        self.variaional = variaional 
        self.sigma      = sigma
        self.latent_dim = latent_dim
        
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
        if "first_layer" in self.config["conv"] and self.config["conv"]["first_layer"]:
            conv_modules_list.append(self.conv(in_channels = self.tensor_shapes[0][0], out_channels = self.tensor_shapes[1][0], kernel_size = 3, stride = 1, padding = 1) )
            range_ = [1, self.config["conv"]["n_blocks"] + 1]
        else:
            range_ = [0, self.config["conv"]["n_blocks"]]
        for i in range(*range_):
            batch_norm = True if "batch_norm" in self.config and self.config["batch_norm"] else False
            conv_modules_list.append(
            Downsample(channels = self.tensor_shapes[i][0], out_channels = self.tensor_shapes[i+1][0], kernel_size = self.config["conv"]["kernel_size"], stride = self.config["conv"]["stride"], padding = self.config["conv"]["padding"], conv_layer = self.conv, batch_norm = batch_norm) 
            )    
            conv_modules_list.append(self.act_func)
            if "upsample" in self.config:
                conv_modules_list.append(nn.MaxPool2d(2, stride=2))
        self.conv_seq = nn.Sequential(*conv_modules_list)
        
        # Add if specified a fully connected layer after the convolutions
        if self.linear:
            #self.lin_layer = nn.Linear(in_features = self.tensor_shapes[self.config["conv"]["n_blocks"] + 1][0], out_features = self.latent_dim)
            if self.sigma:
                self.lin_layer = nn.Linear(in_features = self.tensor_shapes[-2][-1], out_features = self.latent_dim * 2)
            else:
                self.lin_layer = nn.Linear(in_features = self.tensor_shapes[-2][-1], out_features = self.latent_dim)
            # if sigma in the config is true we want to encode a standard deviation too 
            #if self.sigma:
            #    self.lin_sig = nn.Linear(in_features = self.tensor_shapes[self.config["conv"]["n_blocks"] + 1][0], out_features = self.latent_dim)    
        
    def forward(self, x):
        # Apply all modules in the seqence
        x = self.conv_seq(x)
        self.logger.debug("after all conv blocks x.shape: " + str(x.shape))
        if self.variaional:
            self.logger.debug("Shape is x.shape: " + str(x.shape))
            self.logger.debug("tensor Shapes : " + str(self.tensor_shapes))
            if self.linear and self.sigma:
                # maybe absolut value of sigma act ReLu
                self.logger.debug("Shape should be: [" + str(self.config["batch_size"]) + "," + str(self.tensor_shapes[-2][0]) + "]")
                x = x.view(-1, self.tensor_shapes[-2][0])
                self.logger.debug("before linear layer x.shape: " + str(x.shape))            
                x = self.lin_layer(x)
                self.logger.debug("after linear layer x.shape: " + str(x.shape))            
            else:
                self.logger.debug("Shape should be: [" + str(self.config["batch_size"]) + "," + str(self.tensor_shapes[-1][0]) + "]")
                x = x.view(-1, self.tensor_shapes[-1][0])
                self.logger.debug("x.shape: " + str(x.shape))            
        
        #if [x.shape[-2],x.shape[-1]] == [1,1]:
        #    x = x.view(-1,x.shape[1])    
        return x
    
class VAE_Model_Decoder(nn.Module):
    def __init__(
        self,
        config, 
        act_func, 
        tensor_shapes,
        conv = NormConv2d,
        linear     = None,
        variaional = None,
        sigma      = None,
        latent_dim = None
    ):
        super(VAE_Model_Decoder,self).__init__()
        self.logger = get_logger("VAE_Model_Decoder")
        # save all required parameters
        self.config = config
        self.act_func = act_func
        self.tensor_shapes = tensor_shapes
        self.conv       = conv
        self.linear     = linear
        self.variaional = variaional
        self.sigma      = sigma
        self.latent_dim = latent_dim
        
        if self.linear:
            self.lin_layer = nn.Linear(in_features = self.latent_dim, out_features = self.tensor_shapes[self.config["conv"]["n_blocks"] + 1][0])
        
        self.conv_seq = self.get_blocks()
    
    def add_conv(self, add_kernel_size = 0, padding = 0):
        return nn.ConvTranspose2d(in_channels = self.config["conv"]["conv_channels"][0], out_channels = self.config["conv"]["conv_channels"][0], 
                                        kernel_size = self.config["conv"]["kernel_size"] + add_kernel_size, stride = 1, padding = padding)

    def get_blocks(self):
        upsample_modules_list = []
        for i in range(self.config["conv"]["n_blocks"], 0, -1):
            '''
            if "upsample" in self.config:
                if "conv_after_upsample" in self.config and self.config["conv_after_upsample"]:
                    # TODO this has to be fixed somehow
                    print("channels:",self.config["conv"]["conv_channels"][i], "  spacial_size to upsample to:",self.tensor_shapes[i-1][1:])
                    upsample_modules_list.append(self.conv(in_channels = self.config["conv"]["conv_channels"][i], out_channels = self.config["conv"]["conv_channels"][i-1], kernel_size=3, stride=1, padding=1))
                upsample_modules_list.append(nn.Upsample(size = self.tensor_shapes[i-1][1:], mode = self.config["upsample"], conv_layer = self.conv))
            else:
            '''
            if "batch_norm" in self.config and self.config["batch_norm"] and i != 1:
                upsample_modules_list.append(Upsample(in_channels = self.config["conv"]["conv_channels"][i], out_channels = self.config["conv"]["conv_channels"][i-1], conv_layer = self.conv, batch_norm = True))
            else:
                upsample_modules_list.append(Upsample(in_channels = self.config["conv"]["conv_channels"][i], out_channels = self.config["conv"]["conv_channels"][i-1], conv_layer = self.conv))
            if i != 1:
                upsample_modules_list.append(self.act_func)
            else:
                if "final_layer" in self.config["conv"] and self.config["conv"]["final_layer"]:
                    if "final_layer_kernel_size" in self.config["conv"]:
                        kernel_size = self.config["conv"]["final_layer_kernel_size"]
                        if kernel_size%2 == 1:
                            padding = int(kernel_size//2)
                        elif kernel_size%2 == 0:
                            padding = int((kernel_size/2)-1)
                            upsample_modules_list.append(One_sided_padding())
                    else:
                        kernel_size = 1
                        padding = 0
                    upsample_modules_list.append(self.conv(self.config["conv"]["conv_channels"][0], self.config["conv"]["conv_channels"][0], kernel_size=kernel_size, padding = padding))
                upsample_modules_list.append(nn.Tanh())
        return nn.Sequential(*upsample_modules_list)

    def forward(self, x):
        if self.linear:
            x = self.act_func(self.lin_layer(x))
        if self.variaional and not self.linear:
            x = x.reshape(-1,*self.tensor_shapes[-2])
        elif self.variaional and self.linear:
            x = x.reshape(-1,*self.tensor_shapes[-3])
        x = self.conv_seq(x)
        return x

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