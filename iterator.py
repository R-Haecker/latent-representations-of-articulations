import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import numpy as np
import torch
import torch.nn as nn
from edflow import TemplateIterator, get_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Iterator(TemplateIterator):
    def __init__(self, config, root, model, *args, **kwargs):
        super().__init__(config, root, model, *args, **kwargs)
        self.config = config
        self.logger = get_logger("Iterator")
        self.check_config(config)
        model = model.to(device)
        
        # loss and optimizer
        self.criterion = self.get_loss_funct(config["loss_function"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])

        if "track_loss" in self.config:
            self.track_criterion = self.get_loss_funct(self.config["track_loss"])
        

    def check_config(self, config):
        assert "loss_function" in config, "The config must contain and define a Loss function. possibilities:{'L1','L2'or'MSE','KL'or'KLD'}."
        assert "learning_rate" in config, "The config must contain and define a the learning rate."
        assert "weight_decay" in config, "The config must contain and define a the weight decay."
        
    def get_loss_funct(self, loss_function):
        if loss_function == "L1":
            return nn.L1Loss()
        if loss_function == "L2" or loss_function == "MSE":
            return nn.MSELoss()
        if loss_function == "KL" or loss_function == "KLD":
            return nn.KLDivLoss()

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
        inputs = torch.tensor(inputs).to(device)
        
        # compute loss
        results = model(inputs)
        if type(results)==tuple:
            outputs, mu , var = results
        else:
            outputs = results

        self.logger.debug("inputs.shape: " + str(inputs.shape))
        self.logger.debug("output.shape: " + str(outputs.shape))
        
        loss = self.criterion(outputs, inputs)
        mean_loss = torch.mean(loss)
        min_loss = np.min(loss.cpu().detach().numpy())
        max_loss = np.max(loss.cpu().detach().numpy())    
        
        if "track_loss" in self.config:
            track_loss = self.track_criterion(outputs, inputs)
            track_mean_loss = torch.mean(track_loss)
            track_min_loss = np.min(track_loss.cpu().detach().numpy())
            track_max_loss = np.max(track_loss.cpu().detach().numpy())    
            log_dict = {
                "images": {"inputs": inputs.cpu().detach().permute(0,2,3,1).numpy(),"outputs": outputs.cpu().detach().permute(0,2,3,1).numpy()},
                "scalars": 
                {
                    self.config["loss_function"]:
                    {
                        "min_loss": min_loss,
                        "max_loss": max_loss,
                        "mean_loss": mean_loss
                    },
                    self.config["track_loss"]:
                    {
                        "min_loss": track_min_loss,
                        "max_loss": track_max_loss,
                        "mean_loss": track_mean_loss
                    }
                }
            }
        else:
            log_dict = {
                "images": {"inputs": inputs.cpu().detach().permute(0,2,3,1).numpy(),"outputs": outputs.cpu().detach().permute(0,2,3,1).numpy()},
                "scalars": 
                {
                    self.config["loss_function"]:
                    {
                        "min_loss": min_loss,
                        "max_loss": max_loss,
                        "mean_loss": mean_loss
                    }    
                }
            }

        def train_op():
            self.optimizer.zero_grad()
            mean_loss.backward()
            self.optimizer.step()
            del log_dict["images"]

        def log_op():
            return log_dict

        def eval_op():
            return {}
        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

#TODO look at optimzer in config and how to choose their parameters
#TODO inside iterator init all interesting thing logging send to wandb