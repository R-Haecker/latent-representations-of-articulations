import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

#import edflow.custom_logging
from edflow import get_logger
from edflow.custom_logging import LogSingleton
import numpy as np

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        LogSingleton.set_log_level("debug")
        self.logger = get_logger("Model")
        self.config = config
        self.check_config()
        self.tensor_shapes = self.get_tensor_shapes()
        self.logger.info(str(self.tensor_shapes))

        self.act_func = self.get_act_func()
        self.enc = encoder(config = config, act_func = self.act_func, tensor_shapes = self.tensor_shapes)
        self.dec = decoder(config = config, act_func = self.act_func, tensor_shapes = self.tensor_shapes)
    
    def sample_gaus(self, mu, sig = 1):
        assert "variational" in self.config, "If you want to sample from a gaussian distribution to create images you need the key 'variational' in the config."
        if type(mu) in [int,float]:
            mu = torch.ones([self.config["linear"]["latent_dim"]]) * mu
            sig = torch.ones([self.config["linear"]["latent_dim"]]) * sig
        if "sigma" in self.config["variational"] and self.config["variational"]["sigma"]: 
            x = [mu, sigma]
        else:
            x = mu
        
        x = self.dec(x)
        return x
        
    def forward(self, x):
        z = self.enc(x)
        self.logger.debug("encoder output: " + str(z.shape))
        x = self.dec(z)
        self.z = self.dec.z
        self.logger.debug("decoder output: " + str(x.shape))
        #print("decoder output: " + str(x.shape))
        return x 

    def get_tensor_shapes(self):
        """This member function calculates the shape of a every tensor after an operation.        
        :return: A list of the shape of an tensors after every module.
        :rtype: List
        """        
        tensor_shapes = []
        # The first shape is specified in the config
        tensor_shapes.append([3, self.config["image_resolution"][0],self.config["image_resolution"][1]])
        # calculate the shape after a convolutuonal operation
        if ([ self.config["conv"]["kernel_size"], self.config["conv"]["stride"] ] in [[4,3],[6,3]] or ("upsample" in self.config and [ self.config["conv"]["kernel_size"], self.config["conv"]["stride"] ] in [[2,2],[3,3],[4,4]])):
            # these cases just have one less spacial dimension in the last iteration...  
            for i in range(self.config["conv"]["amount_conv_blocks"]): 
                # calculate the spacial resolution
                spacial_res = (int((tensor_shapes[i][-1] + 2 * self.config["conv"]["padding"] - self.config["conv"]["kernel_size"] - 2)
                            /self.config["conv"]["stride"] + 1) + 1)
                if "upsample" in self.config:
                    spacial_res = int(np.floor(spacial_res/2) +1) 
                if i == self.config["conv"]["amount_conv_blocks"]-1:
                    spacial_res = int(spacial_res - 1 )
                tensor_shapes.append([self.config["conv"]["conv_channels"][i+1], spacial_res, spacial_res])    
        else:
            # normal formular to compute the tensor shapes     
            for i in range(self.config["conv"]["amount_conv_blocks"]): 
                # calculate the spacial resolution
                spacial_res = (int((tensor_shapes[i][-1] + 2 * self.config["conv"]["padding"] - self.config["conv"]["kernel_size"] - 2)
                            /self.config["conv"]["stride"] + 1) + 1)
                if "upsample" in self.config:
                    spacial_res = int(np.floor(spacial_res/2) +1) 
                tensor_shapes.append([self.config["conv"]["conv_channels"][i+1], spacial_res, spacial_res])
        
        # add the shape of the flatten image if a fc linaer layer is available
        if "linear" in self.config:
            if "latent_dim" in self.config["linear"] and not self.config["linear"]["latent_dim"] == 0:
                flatten_rep = tensor_shapes[self.config["conv"]["amount_conv_blocks"]][1] * tensor_shapes[self.config["conv"]["amount_conv_blocks"]][2] * self.config["conv"]["conv_channels"][self.config["conv"]["amount_conv_blocks"]]
                tensor_shapes.append([flatten_rep])
                tensor_shapes.append([self.config["linear"]["latent_dim"]])
        return tensor_shapes

    def check_config(self):
        assert "activation_function" in self.config, "For this model you need to specify the activation function: possible options :{'ReLU, LeakyReLu, Sigmoid, LogSigmoid, Tanh, SoftMax'}"
        assert "image_resolution" in self.config, "You have to specify the resolution of the images which are given to the model."
        assert "kernel_size" in self.config["conv"], "For this convolutional model you have to specify the kernel size of all convolutions."
        assert "stride" in self.config["conv"], "For this convolutional model you have to specify the stride value of all convolutions."
        
        assert "conv_channels" in self.config["conv"], "The amount of channels at every convolution have to be specified in the config nested in 'conv' at 'conv_channels'."
        if not self.config["conv"]["conv_channels"][0] == 3:
            self.config["conv"]["conv_channels"].insert(0, 3)
            self.logger.info("In config the first conv chanlles should always be three. It is now added.")
        if "amount_conv_blocks" not in self.config:
            self.config["conv"]["amount_conv_blocks"] = len(self.config["conv"]["conv_channels"]) - 1 
        else:
            assert len(self.config["conv"]["conv_channels"]) == self.config["conv"]["amount_conv_blocks"]+1, "The first conv_chanel is now three --> The amount of convolutional blocks: 'amount_conv_blocks' = " + str(self.config["conv"]["amount_conv_blocks"]) + " plus one has to be the same as the length of the 'conv_channels' list: len('conv_channels') = " + str(len(self.config["conv"]["conv_channels"]))
        if "padding" not in self.config["conv"] or "upsample" in self.config:
                if self.config["conv"]["kernel_size"]%2 == 0:
                    self.config["conv"]["padding"] = 0
                elif self.config["conv"]["kernel_size"] == 1:
                    self.config["conv"]["padding"] = 0
                elif self.config["conv"]["kernel_size"] == 3:
                    self.config["conv"]["padding"] = 1
                elif self.config["conv"]["kernel_size"] == 5:
                    self.config["conv"]["padding"] = 2
                else:
                    assert 1==0, "Kernal size is too big."
                self.logger.debug("Padding of the convolutions is set to " + str(self.config["conv"]["padding"]))
        
        #if "upsample" in self.config:
            #assert self.config["conv"]["stride"] == 1, "For now if you want to upsample/downsample you can only use convolutions wich keep the spacial size the same. You have to use stride = 1."
            #assert self.config["conv"]["kernel_size"]%2 == 1, "For now if you want to upsample/downsample you can only use convolutions wich keep the spacial size the same. You have to use an odd kernal size."
        if "variational" in self.config:
            assert "linear" in self.config and "latent_dim" in self.config["linear"] and self.config["linear"]["latent_dim"] != 0, "If you want to use a variational auto encoder you have to use a linear layer at the bottleneck. Specify 'linear' and within 'latent_dim' in the config wich has to be none zero."

    def get_act_func(self):
        if self.config["activation_function"] == "ReLU":
            if "ReLU" in self.config:
                self.logger.debug("activation function: changed ReLu to leakyReLU with secified slope!")
                return nn.LeakyReLU(negative_slope=self.config["ReLu"])
            else:
                self.logger.debug("activation function: ReLu")
                return nn.ReLU(True)  
        if self.config["activation_function"] == "LeakyReLU":
            if "LeakyReLU_negative_slope" in self.config:
                self.logger.debug("activation_function: LeakyReLU")
                return nn.LeakyReLU(negative_slope=self.config["LeakyReLU_negative_slope"])
            elif "LeakyReLU" in self.config:
                self.logger.debug("activation_function: LeakyReLU")
                return nn.LeakyReLU(negative_slope=self.config["LeakyReLU"])
            else:
                self.logger.debug("activation function: LeakyReLu changed to ReLU because no slope value could be found")
                return nn.LeakyReLU()
        if self.config["activation_function"] == "Sigmoid":
            self.logger.debug("activation_function: Sigmoid")
            return nn.Sigmoid
        if self.config["activation_function"] == "LogSigmoid":
            self.logger.debug("activation_function: LogSigmoid")
            return nn.LogSigmoid
        if self.config["activation_function"] == "Tanh":
            self.logger.debug("activation_function: Tanh")
            return nn.Tanh
        if self.config["activation_function"] == "SoftMax":
            self.logger.debug("activation_function: SoftMax")
            return nn.SoftMax()

class encoder(nn.Module):
    def __init__(self, config, act_func, tensor_shapes):
        super(encoder,self).__init__()
        self.act_func = act_func
        self.config = config 
        self.logger = get_logger("Model_encoder")
        # Get the shapes of the tensors in this model
        self.tensor_shapes = tensor_shapes
        # Add convolutional blocks 
        self.conv_seq = self.get_blocks()
        # Add if specified a fully connected layer after the convolutions
        if "linear" in self.config:
            if "latent_dim" in self.config["linear"] and not self.config["linear"]["latent_dim"] == 0:
                if "variational" in self.config and "sigma" in self.config["variational"] and self.config["variational"]["sigma"]:
                    self.lin_mu = nn.Linear(in_features = self.tensor_shapes[self.config["conv"]["amount_conv_blocks"] + 1][0], out_features = self.config["linear"]["latent_dim"])
                    self.lin_sig = nn.Linear(in_features = self.tensor_shapes[self.config["conv"]["amount_conv_blocks"] + 1][0], out_features = self.config["linear"]["latent_dim"])
                else:
                    self.lin_layer = nn.Linear(in_features = self.tensor_shapes[self.config["conv"]["amount_conv_blocks"] + 1][0], out_features = self.config["linear"]["latent_dim"])

    def get_blocks(self):
        """Create the convolutional blocks specified in the config. 
        
        :return: A Sequentail with all convolutiuonal modules.
        :rtype: nn.Sequential
        """        
        conv_modules_list = []
        if "upsample" in self.config:
            for i in range(self.config["conv"]["amount_conv_blocks"]):
                conv_modules_list.append( 
                    nn.Conv2d(in_channels = self.config["conv"]["conv_channels"][i], out_channels = self.config["conv"]["conv_channels"][i+1], kernel_size = self.config["conv"]["kernel_size"], stride = self.config["conv"]["stride"], padding = self.config["conv"]["padding"])
                    )
                conv_modules_list.append(self.act_func)
                conv_modules_list.append(nn.MaxPool2d(2, stride=2))
        else:
            for i in range(self.config["conv"]["amount_conv_blocks"]):
                conv_modules_list.append( 
                    nn.Conv2d(in_channels = self.config["conv"]["conv_channels"][i], out_channels = self.config["conv"]["conv_channels"][i+1], kernel_size = self.config["conv"]["kernel_size"], stride = self.config["conv"]["stride"], padding = self.config["conv"]["padding"])
                    )
                conv_modules_list.append(self.act_func)
        return nn.Sequential(*conv_modules_list)
        
    def forward(self,x):
        # Apply all modules in the seqence
        x = self.conv_seq(x)
        self.logger.debug("after all conv blocks x.shape: " + str(x.shape))
        if "variational" in self.config:
            if "sigma" in self.config["variational"] and self.config["variational"]["sigma"]:
                x = x.view(-1, self.tensor_shapes[self.config["conv"]["amount_conv_blocks"] + 1][0])
                self.logger.debug("Shape should be: [" + str(config["batch_size"]) + "," + str(self.tensor_shapes[self.config["conv"]["amount_conv_blocks"] + 1][0]) + "]")
                self.logger.debug("Shape is x.shape: " + str(c.shape))
                # maybe absolut value of sigma act ReLu
                x = [self.lin_mu(x), torch.abs(self.lin_sig(x))]
            else:
                self.logger.debug("Shape should be: [" + str(self.config["batch_size"]) + "," + str(self.tensor_shapes[self.config["conv"]["amount_conv_blocks"] + 1][0]) + "]")
                self.logger.debug("Shape is x.shape: " + str(x.shape))
                x = x.view(-1, self.tensor_shapes[self.config["conv"]["amount_conv_blocks"] + 1][0])
                self.logger.debug("Shape is x.shape: " + str(x.shape))
                x = self.lin_layer(x)

        elif "linear" in self.config and "latent_dim" in self.config["linear"] and not self.config["linear"]["latent_dim"] == 0:
            # If specified: Add a fully connected linear layer after the blocks to downsample to a specific bottleneck size
            # flatten the image representation
            self.logger.debug("befor Linear layer: x.shape == " + str(x.shape))
            #print(("befor Linear layer: x.shape == " + str(x.shape)))
            x = x.view(-1, self.tensor_shapes[self.config["conv"]["amount_conv_blocks"] + 1][0])
            self.logger.debug("befor Linear layer: x.shape == " + str(x.shape))
            #print("befor Linear layer: x.shape == " + str(x.shape))
            # use fc layer and the activation function
            x = self.lin_layer(x)
            x = self.act_func(x)        
        return x
    
class decoder(nn.Module):
    def __init__(self, config, act_func, tensor_shapes):
        super(decoder,self).__init__()
        self.config = config
        self.logger = get_logger("Model_decoder")
        self.act_func = act_func
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tensor_shapes = tensor_shapes
        if "linear" in self.config:
            if "latent_dim" in self.config["linear"] and not self.config["linear"]["latent_dim"] == 0:
                self.lin_layer = nn.Linear(in_features = self.config["linear"]["latent_dim"], out_features = self.tensor_shapes[self.config["conv"]["amount_conv_blocks"] + 1][0])
        self.conv_seq = self.get_blocks()

        self.Tanh = nn.Tanh()
    
    def add_conv(self, add_kernel_size = 0, padding = 0):
        return nn.ConvTranspose2d(in_channels = self.config["conv"]["conv_channels"][0], out_channels = self.config["conv"]["conv_channels"][0], 
                                        kernel_size = self.config["conv"]["kernel_size"] + add_kernel_size, stride = 1, padding = padding)

    def get_blocks(self):
        # Get convolutional block with fitting spacial size
        # downsampling with convolutions
        convT_modules_list = []
        for i in range(self.config["conv"]["amount_conv_blocks"], 0, -1):
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
        if "linear" in self.config:
            if "latent_dim" in self.config["linear"] and not self.config["linear"]["latent_dim"] == 0:
                if "variational" in self.config:
                    norm_dist = torch.distributions.normal.Normal(torch.zeros([self.config["linear"]["latent_dim"]]), torch.ones([self.config["linear"]["latent_dim"]]))
                    eps = norm_dist.sample().to(self.device)#.double()
                    if "sigma" in self.config["variational"] and self.config["variational"]["sigma"]:
                        mu = x[0]
                        var = x[1]
                        z = mu + var * eps
                    else:
                        #print("mu.shape:",x.shape)
                        #print("type(mu):",type(x))
                        #print("mu:",x)
                        mu = x
                        z = mu + eps
                else:        
                    z = x
                #print("anything")
                self.z = z
                x = self.act_func(self.lin_layer(z))
                x = x.reshape(-1,*self.tensor_shapes[self.config["conv"]["amount_conv_blocks"]])
        else:
            self.z = x
        x = self.Tanh(self.conv_seq(x))
        #x = torch.reshape(x, (self.config["batch_size"],3,self.config["image_resolution"][0],self.config["image_resolution"][1]))
        return x

#TODO log the spacial sizes everywhere in wandb

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


#TODO get kl loss after a model is trained for a while
#TODO late dev look for alternatives or toggling maxpool2d
