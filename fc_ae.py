import torch
import torchvision
import torch.nn as nn

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
        assert "activation_function" in config, "For this fully linear connected model you need to specify the activation function: possible options :{'ReLU, LeakyReLu, Sigmoid, LogSigmoid, Tanh, SoftMax'}"
        assert "image_resolution" in config, "You have to specify the resolution of the images which are given to the model."
        if "hidden_layer_multiplicator" not in config:
            assert "hidden_layer_size" in config, "For this model with only fully connected layer you have to specify how many neurons are in the hidden layers."
        else:
            config["hidden_layer_size"] = int(config["image_resolution"][0]*config["image_resolution"][1]*3*config["hidden_layer_multiplicator"]*config["batch_size"])
        assert "latent_dim" in config, "For this model with only fully connected layer you have to specify how many numbers represent the bottleneck."

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
        if config["hidden_layer_size"]==0:
            self.fc = nn.Linear(in_features = config["image_resolution"][0]*config["image_resolution"][1]*3, out_features = config["latent_dim"]) 
        else:
            self.fc1 = nn.Linear(in_features = config["image_resolution"][0]*config["image_resolution"][1]*3, out_features = config["hidden_layer_size"])
            self.fc2 = nn.Linear(in_features = config["hidden_layer_size"], out_features= config["latent_dim"])

    def forward(self,x):
        x = x.view(self.config["batch_size"],-1)
        if self.config["hidden_layer_size"]==0:
            x = self.act_func(self.fc(x))
        else:
            x = self.act_func(self.fc1(x))
            x = self.act_func(self.fc2(x))
        return x
    
class decoder(nn.Module):
    def __init__(self, config, act_func):
        super(decoder,self).__init__()
        self.config = config
        self.act_func = act_func
        self.Tanh = nn.Tanh()
        if config["hidden_layer_size"]==0:
            self.fc = nn.Linear(in_features = config["latent_dim"], out_features = config["image_resolution"][0]*config["image_resolution"][1]*3)
            self.hidden_layer = False
        else:
            self.fc1 = nn.Linear(in_features = config["latent_dim"], out_features = config["hidden_layer_size"])
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

#TODO find out what has to be send to cudda to make i work