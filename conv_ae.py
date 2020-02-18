import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from edflow import get_logger
import numpy as np

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.logger = get_logger("Model")
        self.config = config
        self.check_config(config)
        self.tensor_shapes = self.get_tensor_shapes()
        #print(self.tensor_shapes)
        self.act_func = self.get_act_func(config)
        self.enc = encoder(config = config, act_func = self.act_func, tensor_shapes = self.tensor_shapes)
        self.dec = decoder(config = config, act_func = self.act_func, tensor_shapes = self.tensor_shapes)
    
    def forward(self, x):
        z = self.enc(x)
        #print("tensor_shapes:", self.enc.tensor_shapes)
        #print("encoder output:", z.shape)
        x = self.dec(z)
        #print("decoder output:", x.shape)
        return x 

    def get_tensor_shapes(self):
        """This member function calculates the shape of a every tensor after an operation.        
        :return: A list of the shape of an tensors after every module.
        :rtype: List
        """        
        tensor_shapes = []
        # The first shape is specified in the config
        tensor_shapes.append([self.config["batch_size"], 3, self.config["image_resolution"][0],self.config["image_resolution"][1]])
        # calculate the shape after a convolutuonal operation
        if (self.config["conv"]["kernel_size"] == 4 and self.config["conv"]["stride"] == 3) or (self.config["conv"]["kernel_size"] == 6 and self.config["conv"]["stride"] == 3):
            for i in range(self.config["conv"]["amount_conv_blocks"]): 
                # calculate the spacial resolution
                spacial_res = (int((tensor_shapes[i][-1] + 2 * self.config["conv"]["padding"] - self.config["conv"]["kernel_size"] - 2)
                            /self.config["conv"]["stride"] + 1) + 1)
                if i == self.config["conv"]["amount_conv_blocks"]-1:
                    spacial_res += -1 
                if "upsample" in self.config:
                    spacial_res = (spacial_res/2)
                tensor_shapes.append([self.config["batch_size"], self.config["conv"]["conv_channels"][i+1], spacial_res, spacial_res])    
        else:    
            for i in range(self.config["conv"]["amount_conv_blocks"]): 
                # calculate the spacial resolution
                spacial_res = (int((tensor_shapes[i][-1] + 2 * self.config["conv"]["padding"] - self.config["conv"]["kernel_size"] - 2)
                            /self.config["conv"]["stride"] + 1) + 1)
                if "upsample" in self.config:
                    spacial_res = int(np.floor(spacial_res/2) +1) 
                tensor_shapes.append([self.config["batch_size"], self.config["conv"]["conv_channels"][i+1], spacial_res, spacial_res])
        # add the shape of the flatten image if a fc linaer layer is available
        if "linear" in self.config:
            if "bottleneck_size" in self.config["linear"] and not self.config["linear"]["bottleneck_size"] == 0:
                flatten_rep = tensor_shapes[self.config["conv"]["amount_conv_blocks"]][2] * tensor_shapes[self.config["conv"]["amount_conv_blocks"]][3] * self.config["conv"]["conv_channels"][self.config["conv"]["amount_conv_blocks"]]
                tensor_shapes.append([self.config["batch_size"], flatten_rep])
                tensor_shapes.append([self.config["batch_size"], self.config["linear"]["bottleneck_size"]])
        return tensor_shapes

    def check_config(self, config):
        assert "activation_function" in config, "For this model you need to specify the activation function: possible options :{'ReLU, LeakyReLu, Sigmoid, LogSigmoid, Tanh, SoftMax'}"
        assert "image_resolution" in config, "You have to specify the resolution of the images which are given to the model."
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
        if "padding" not in self.config["conv"]:
                if self.config["conv"]["kernel_size"]%2 == 0:
                    self.config["conv"]["padding"] = 0
                else:
                    self.config["conv"]["padding"] = 1
        #not allowing following configs
        #assert not (self.config["conv"]["kernel_size"] == 4 and self.config["conv"]["stride"] == 3), "not allowing this config."
  
    
    def get_act_func(self, config):
        if config["activation_function"] == "ReLU":
            if "ReLU" in config:
                self.logger.debug("activation function: changed ReLu to leakyReLU with secified slope!")
                return nn.LeakyReLU(negative_slope=config["ReLu"])
            else:
                self.logger.debug("activation function: ReLu")
                return nn.ReLU(True)  
        if config["activation_function"] == "LeakyReLU":
            if "LeakyReLU_negative_slope" in config:
                self.logger.debug("activation_function: LeakyReLU")
                return nn.LeakyReLU(negative_slope=config["LeakyReLU_negative_slope"])
            elif "LeakyReLU" in config:
                self.logger.debug("activation_function: LeakyReLU")
                return nn.LeakyReLU(negative_slope=config["LeakyReLU"])
            else:
                self.logger.debug("activation function: LeakyReLu changed to ReLU because no slope value could be found")
                return nn.LeakyReLU()
        if config["activation_function"] == "Sigmoid":
            self.logger.debug("activation_function: Sigmoid")
            return nn.Sigmoid
        if config["activation_function"] == "LogSigmoid":
            self.logger.debug("activation_function: LogSigmoid")
            return nn.LogSigmoid
        if config["activation_function"] == "Tanh":
            self.logger.debug("activation_function: Tanh")
            return nn.Tanh
        if config["activation_function"] == "SoftMax":
            self.logger.debug("activation_function: SoftMax")
            return nn.SoftMax()

class encoder(nn.Module):
    def __init__(self, config, act_func, tensor_shapes):
        super(encoder,self).__init__()
        self.act_func = act_func
        self.config = config 
        # Get the shapes of the tensors in this model
        self.tensor_shapes = tensor_shapes
        # Add convolutional blocks 
        self.conv_seq = self.get_blocks()
        # Add if specified a fully connected layer after the convolutions
        if "linear" in self.config:
            if "bottleneck_size" in self.config["linear"] and not self.config["linear"]["bottleneck_size"] == 0:
                self.lin_layer = nn.Linear(in_features = self.tensor_shapes[self.config["conv"]["amount_conv_blocks"] + 1][1], out_features = self.config["linear"]["bottleneck_size"])

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
        # If specified: Add a fully connected linear layer after the blocks to downsample to a specific bottleneck size
        if "linear" in self.config:
            if "bottleneck_size" in self.config["linear"] and not self.config["linear"]["bottleneck_size"] == 0:
                # flatten the image representation
                #print("x.shape:",x.shape)
                x = x.view(self.config["batch_size"], -1)
                # use fc layer and the activation function
                x = self.lin_layer(x)
                x = self.act_func(x)        
        return x
    
class decoder(nn.Module):
    def __init__(self, config, act_func, tensor_shapes):
        super(decoder,self).__init__()
        self.config = config
        self.act_func = act_func
        self.enc_tensor_shapes = tensor_shapes
        if "linear" in self.config:
            if "bottleneck_size" in self.config["linear"] and not self.config["linear"]["bottleneck_size"] == 0:
                self.lin_layer = nn.Linear(in_features =  self.config["linear"]["bottleneck_size"], out_features = self.enc_tensor_shapes[self.config["conv"]["amount_conv_blocks"] + 1][1])
        self.conv_seq = self.get_blocks()

        self.Tanh = nn.Tanh()
    
    def get_conv(self, padding, output_padding):
        convT_modules_list = []
        if type(padding) == list:
            for i in range(self.config["conv"]["amount_conv_blocks"], 0, -1):
                    if i != 1:    
                        if "upsample" in self.config:
                            convT_modules_list.append(nn.Upsample(size = self.enc_tensor_shapes[i-1][2:], mode = self.config["upsample"]))
                        convT_modules_list.append(
                                nn.ConvTranspose2d(in_channels = self.config["conv"]["conv_channels"][i], out_channels = self.config["conv"]["conv_channels"][i-1], 
                                                    kernel_size = self.config["conv"]["kernel_size"], stride = self.config["conv"]["stride"], 
                                                    padding = padding[0],
                                                    output_padding = output_padding[0])
                                )
                        convT_modules_list.append(self.act_func)
                    else:
                        if "upsample" in self.config:
                            convT_modules_list.append(nn.Upsample(size = self.enc_tensor_shapes[i-1][2:], mode = self.config["upsample"]))
                        convT_modules_list.append(
                            nn.ConvTranspose2d(in_channels = self.config["conv"]["conv_channels"][i], out_channels = self.config["conv"]["conv_channels"][i-1], 
                                                kernel_size = self.config["conv"]["kernel_size"], stride = self.config["conv"]["stride"], 
                                                padding = padding[1],
                                                output_padding = output_padding[1])
                            )
        else:
            for i in range(self.config["conv"]["amount_conv_blocks"], 0, -1):
                if "upsample" in self.config:
                    if i != 1:        
                        convT_modules_list.append(nn.Upsample(size = self.enc_tensor_shapes[i-1][2:], mode = self.config["upsample"]))
                    else:
                        convT_modules_list.append(nn.Upsample(size = [self.enc_tensor_shapes[i-1][2]-self.config["conv"]["kernel_size"] + 1 ,self.enc_tensor_shapes[i-1][2]-self.config["conv"]["kernel_size"] + 1], mode = self.config["upsample"]))
                
                convT_modules_list.append(
                        nn.ConvTranspose2d(in_channels = self.config["conv"]["conv_channels"][i], out_channels = self.config["conv"]["conv_channels"][i-1], 
                                            kernel_size = self.config["conv"]["kernel_size"], stride = self.config["conv"]["stride"], 
                                            padding = padding,
                                            output_padding = output_padding)
                        )
                if i != 1:
                    convT_modules_list.append(self.act_func)
        return convT_modules_list

    def get_blocks(self):
            # Get convolutional block with fitting spacial size
            # downsampling with convolutions
            if "upsample" in self.config:
                convT_modules_list = self.get_conv(0,0)
            elif self.config["conv"]["kernel_size"] == 2 and self.config["conv"]["stride"] == 2:  
                convT_modules_list = self.get_conv(0,0)
            elif self.config["conv"]["kernel_size"] == 3 and self.config["conv"]["stride"] == 2:    
                convT_modules_list = self.get_conv(1,1)
            elif self.config["conv"]["kernel_size"] == 4 and self.config["conv"]["stride"] == 2:
                convT_modules_list = self.get_conv([0,0],[1,0])     
            elif self.config["conv"]["kernel_size"] == 5 and self.config["conv"]["stride"] == 2 or self.config["conv"]["kernel_size"] == 5 and self.config["conv"]["stride"] == 3:    
                convT_modules_list = self.get_conv([1,1],[0,1])
            elif self.config["conv"]["kernel_size"] == 3 and self.config["conv"]["stride"] == 3:
                convT_modules_list = self.get_conv(1,0)
            elif self.config["conv"]["kernel_size"] == 4 and self.config["conv"]["stride"] == 3:
                convT_modules_list = self.get_conv([0,0],[2,0])
            elif self.config["conv"]["kernel_size"] == 6 and self.config["conv"]["stride"] == 3:
                convT_modules_list = self.get_conv([0,0],[2,1])
            else:
                convT_modules_list = self.get_conv(0,0)

            return nn.Sequential(*convT_modules_list)    

    def forward(self, x):
        if "linear" in self.config:
            if "bottleneck_size" in self.config["linear"] and not self.config["linear"]["bottleneck_size"] == 0:
                x = self.act_func(self.lin_layer(x))
                x = x.reshape(*self.enc_tensor_shapes[self.config["conv"]["amount_conv_blocks"]])
        x = self.Tanh(self.conv_seq(x))
        #x = torch.reshape(x, (self.config["batch_size"],3,self.config["image_resolution"][0],self.config["image_resolution"][1]))
        return x


#############################
#########  Testing  #########
#############################
'''

import yaml

with open("conv_config.yaml") as fh:
    config_y = yaml.full_load(fh)    

ae = Model(config = config_y)
x = torch.zeros([5,3,64,64])
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


#TODO make get conv shape for all shape not only spacial
#TODO late dev look for alternatives or toggling maxpool2d
#TODO check if function to get padding works for all strides, it depends if the spacial res is odd or not... 
#TODO now do only conv olution settings which keep the spacial resolution 
