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
        self.check_config(config)
        self.act_func = self.get_act_func(config)
        self.enc = encoder(config = config, act_func = self.act_func)
        self.dec = decoder(config = config, act_func = self.act_func)
    
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)
    
    def check_config(self, config):
        assert "activation_function" in config, "For this model you need to specify the activation function: possible options :{'ReLU, LeakyReLu, Sigmoid, LogSigmoid, Tanh, SoftMax'}"
        assert "image_resolution" in config, "You have to specify the resolution of the images which are given to the model."
        assert "kernel_size" in self.config["conv"], "For this convolutional model you have to specify the kernel size of all convolutions."
        assert "stride" in self.config["conv"], "For this convolutional model you have to specify the stride value of all convolutions."
        if "amount_conv_blocks" in self.config:
            assert "conv_channels" in self.config["conv"], "The amount of channels at every convolution have to be specified in the config nested in 'conv' at 'conv_channels'."
            if self.config["conv"]["conv_channels"][0] == 3:
                assert len(self.config["conv"]["conv_channels"]) == self.config["conv"]["amount_conv_blocks"]-1, "You have specified the first conv_chanel as three --> The amount of convolutional blocks: 'amount_conv_blocks' = " + str(self.config["conv"]["amount_conv_blocks"]) + " has to be the same as the length of the 'conv_channels' list: len('conv_channels') = " + str(len(self.config["conv"]["conv_channels"]))     
            else:
                assert len(self.config["conv"]["conv_channels"]) == self.config["conv"]["amount_conv_blocks"], "The amount of convolutional blocks (starting with one): 'amount_conv_blocks' has to be the same as the length of the 'conv_channels' list."
        

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
    def __init__(self, config, act_func):
        super(encoder,self).__init__()
        self.act_func = act_func
        self.config = config 
        
        self.conv_seq_list = self.get_conv_blocks()
        
    def get_padding():
        # with stride one
        if (self.config["conv"]["kernel_size"]/2)%2 == 0:
            return self.config["conv"]["kernel_size"]/2 - 1
        else:
            return np.floor(self.config["conv"]["kernel_size"]/2)

    def get_conv_blocks(self):
        conv_seq_list = []
        padding = get_padding()
        if "amount_conv_blocks" in self.config:
            assert "conv_channels" in self.config["conv"], "The amount of channels at every convolution have to be specified in the config nested in 'conv' at 'conv_channels'."
            if self.config["conv"]["conv_channels"][0] == 3:
                for i in range(self.config["amount_conv_blocks"]):
                    conv_seq_list.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels = self.config["conv"]["conv_channels"][i], out_channels = self.config["conv"]["conv_channels"][i+1], kernel_size = self.config["conv"]["kernel_size"], stride = self.config["conv"]["stride"], padding = padding),
                            self.act_func(),
                            # maybe change this with config later
                            nn.MaxPool2d(2, stride=2)
                        )
                    )
            else:
                conv_seq_list.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels = 3, out_channels = self.config["conv"]["conv_channels"][0], kernel_size = self.config["conv"]["kernel_size"], stride = self.config["conv"]["stride"], padding = padding),
                            self.act_func(),
                            nn.MaxPool2d(2, stride=2)
                        )
                    )
                for i in range(self.config["amount_conv_blocks"]-1):
                    conv_seq_list.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels = self.config["conv"]["conv_channels"][i], out_channels = self.config["conv"]["conv_channels"][i+1], kernel_size = self.config["conv"]["kernel_size"], stride = self.config["conv"]["stride"], padding = padding),
                            self.act_func(),
                            nn.MaxPool2d(2, stride=2)
                        )
                    )
        return conv_seq_list
        
    def forward(self,x):
        for i in range(self.config["amount_conv_blocks"]):
            x = self.conv_seq_list[i](x)
        x = x.view(self.config["batch_size"],-1)
        x = F.linear(x.shape[1], self.config["bottleneck_size"])
        x = self.act_func(x)
        return x
    
class decoder(nn.Module):
    def __init__(self, config, act_func):
        super(decoder,self).__init__()
        self.config = config
        self.act_func = act_func
        self.Tanh = nn.Tanh()
        if config["hidden_layer_size"]==0:
            self.fc = nn.Linear(in_features = config["bottleneck_size"], out_features = config["image_resolution"][0]*config["image_resolution"][1]*3)
            self.hidden_layer = False
        else:
            self.fc1 = nn.Linear(in_features = config["bottleneck_size"], out_features = config["hidden_layer_size"])
            self.fc2 = nn.Linear(in_features = config["hidden_layer_size"], out_features = config["image_resolution"][0]*config["image_resolution"][1]*3)
            self.hidden_layer = True
    def forward(self, x):
        if self.config["hidden_layer_size"]==0:
            x = self.Tanh(self.fc(x))
        else:
            x = self.act_func(self.fc1(x))
            x = self.Tanh((self.fc2(x)))
        x = torch.reshape(x, (self.config["batch_size"],3,self.config["image_resolution"][0],self.config["image_resolution"][1]))
        return x

#TODO late dev look for alternatives or toggling maxpool2d
#TODO check if function to get padding works for all strides, it depends if the spacial res is odd or not... 


class old_Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)
        
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels= 9, kernel_size= 5, stride=3, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

