import os
import numpy as np

import torch
import torch.nn as nn

from edflow import TemplateIterator, get_logger
from edflow.hooks.checkpoint_hooks.common import get_latest_checkpoint
from edflow.util import walk

from iterator.util import (
    get_loss_funct,
    np2pt,
    pt2np
) 

class Iterator(TemplateIterator):
    def __init__(self, config, root, model, *args, **kwargs):
        super().__init__(config, root, model, *args, **kwargs)
        # export to the right gpu if specified in the config
        self.set_gpu()
        # get the config and the logger
        self.config = config
        # Config will be tested inside the Model class even for the iterator
        self.logger = get_logger("Iterator")
        # Log the architecture of the model
        self.logger.debug(f"{model}")
        model = model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])

        if "variational" in self.config:
            # get the offset for the sigmoid regulator for the variational part of the loss 
            self.x_offset_KLD = self.offset_KLD_weight()
    
    def set_gpu(self):
        """Move the model to device cuda if available and use the specified GPU"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.debug(f"Model will pushed to the device: {self.device}")
        if "CUDA_VISIBLE_DEVICES" in self.config:
            if type(self.config["CUDA_VISIBLE_DEVICES"]) != str:
                self.config["CUDA_VISIBLE_DEVICES"] = str(self.config["CUDA_VISIBLE_DEVICES"])
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config["CUDA_VISIBLE_DEVICES"]

    def criterion(self, inputs, predictions, model):
        """This function returns a dictionary with all neccesary losses for the model.
        
        :param inputs: Tensor image inputs for the model.
        :type inputs: Torch.Tensor
        :param predictions: Images predictions from the model.
        :type predictions: Torch.Tensor
        :param model: Model on which to train on.
        :return: Dictionary with all losses
        """        
        losses = {}
        # get the reconstruction loss
        reconstruction_loss = get_loss_funct(self.config["losses"]["reconstruction_loss"])
        # compute the reconstruction loss
        recon_loss = reconstruction_loss(predictions, inputs)
        mean_recon_loss = pt2np(torch.mean(recon_loss), permute = False)
        min_recon_loss  = np.min(pt2np(recon_loss, permute = False))
        max_recon_loss  = np.max(pt2np(recon_loss, permute = False))    
        losses["reconstruction_loss_" + self.config["losses"]["reconstruction_loss"]] = {}
        losses["reconstruction_loss_" + self.config["losses"]["reconstruction_loss"]]["min"] = min_recon_loss
        losses["reconstruction_loss_" + self.config["losses"]["reconstruction_loss"]]["max"] = max_recon_loss
        losses["reconstruction_loss_" + self.config["losses"]["reconstruction_loss"]]["mean"] = mean_recon_loss
        # track and only view an additional loss if specified
        if "view_loss" in self.config["losses"]:
            view_criterion = get_loss_funct(self.config["losses"]["view_loss"])
            # calculate a secound reconstruction loss to log if wanted 
            view_loss = view_criterion(predictions, inputs)
            view_mean_loss = pt2np(torch.mean(view_loss), permute = False)
            view_min_loss  = np.min(pt2np(view_loss, permute = False))
            view_max_loss  = np.max(pt2np(view_loss, permute = False))   
            losses["reconstruction_loss_view_" + self.config["losses"]["view_loss"]] = {}
            losses["reconstruction_loss_view_" + self.config["losses"]["view_loss"]]["min"] = view_min_loss
            losses["reconstruction_loss_view_" + self.config["losses"]["view_loss"]]["max"] = view_max_loss
            losses["reconstruction_loss_view_" + self.config["losses"]["view_loss"]]["mean"] = view_mean_loss
        # compute the loss for the variational part
        if "variational" in self.config:
            # Calculate loss for latent representation if the model is a VAE
            step = torch.tensor(self.get_global_step(), dtype = torch.float)
            KLD_weight = self.get_KLD_weight(step).to(self.device)
            losses["KLD_loss"] = {}
            if "sigma" in self.config["variational"] and self.config["variational"]["sigma"]:
                # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                KLD_loss = -0.5 * torch.sum(1 + model.var.pow(2).log() - model.mu.pow(2) - model.var.pow(2))
                KLD_loss_weighted = KLD_weight * KLD_loss 
                losses["KLD_loss"]["mean"] = KLD_loss
            else:
                abs_z = torch.mean(torch.abs(model.z))
                # force absolute value of latent representation (with a regulation factor) into the loss to minimize it
                KLD_loss_weighted = KLD_weight * abs_z
                losses["KLD_loss"]["mean"] = abs_z

            losses["KLD_loss"]["weight"] = KLD_weight
            losses["KLD_loss"]["mean_weighted"] = KLD_loss_weighted 
            total_loss = mean_recon_loss + KLD_loss_weighted
        losses["total_loss"] = total_loss
        return losses        

    def prepare_logs(self, losses, inputs, predictions):
        """Return a log dictionary with all instersting data to log."""
        # create a dictionary to log with all interesting variables 
        logs = {
            "images": {},
            "scalars":{
                **losses
                }
        }
        # log the input and output images
        in_img = pt2np(inputs)
        out_img = pt2np(predictions)
        for i in range(self.config["batch_size"]):
            logs["images"].update({"input_" + str(i): np.expand_dims(in_img[i],0)})
            logs["images"].update({"output_" + str(i): np.expand_dims(out_img[i],0)})

        def conditional_convert2np(log_item):
            if isinstance(log_item, torch.Tensor):
                log_item = log_item.detach().cpu().numpy()
            return log_item
        # convert to numpy
        walk(logs, conditional_convert2np, inplace=True)
        return logs

    def step_op(self, model, **kwargs):
        '''This function will be called every step in the training.'''
        # get inputs
        inputs = kwargs["image"]
        inputs = torch.tensor(inputs).to(self.device)
        
        # compute the output of the model
        self.logger.debug("inputs.shape: " + str(inputs.shape))
        predictions = model(inputs)
        self.logger.debug("output.shape: " + str(predictions.shape))
        
        losses = self.criterion(inputs, predictions, model)
        
        def train_op():
            # This function will be executed if the model is in training mode
            self.optimizer.zero_grad()
            losses["total_loss"].backward()
            self.optimizer.step()

        def log_op():
            # This function will always execute
            logs = self.prepare_logs(losses, inputs, predictions)
            return logs

        def eval_op():
            # This function will be executed if the model is in evaluation mode
            z = pt2np(model.z, permute = False)
            return {"labels": {"latent_rep": z }}
    
        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}
    
    def save(self, checkpoint_path):
        '''Save the weights of the model to the checkpoint_path.'''
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint_path):
        '''Load the weigths of the model from a previous training.'''
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

    def offset_KLD_weight(self):
        '''Return the offset of the sigmoid function for the loss on mu given the parameter from the config.'''
        if "KLD_loss" in self.config["losses"] and "start_step" in self.config["losses"]["KLD_loss"] and "width" in self.config["losses"]["KLD_loss"] and "amplitude" in self.config["losses"]["KLD_loss"]:
            if "start_amplitude" not in self.config["losses"]["KLD_loss"]:
                self.config["losses"]["KLD_loss"]["start_amplitude"] = 1.E-5
            # check for all parameter
            x_offset_KLD = self.config["losses"]["KLD_loss"]["start_step"] - self.config["losses"]["KLD_loss"]["width"]/3 * (np.tan((2*self.config["losses"]["KLD_loss"]["start_amplitude"])/self.config["losses"]["KLD_loss"]["amplitude"] - 1))
        else:
            self.logger.info("parameters for sigmoid regulator are not specified now choosing default.")
            self.config["losses"]["KLD_loss"]["start_step"] = 50000
            self.config["losses"]["KLD_loss"]["start_amplitude"] = 10**(-5) 
            self.config["losses"]["KLD_loss"]["width"] = 2000
            self.config["losses"]["KLD_loss"]["amplitude"] = 0.001
            x_offset_KLD = self.offset_KLD_weight()
        return x_offset_KLD

    def get_KLD_weight(self, step):
        '''Return the sigmoid regulator for the absolute loss of the varaitional expected value mu.'''
        if step < self.config["losses"]["KLD_loss"]["start_step"]:
            return torch.Tensor([0])
        else:
            return self.config["losses"]["KLD_loss"]["amplitude"] * (0.5 + 0.5 * np.tanh((step - self.x_offset_KLD)*3/self.config["losses"]["KLD_loss"]["width"]))

#TODO look at optimzer in config and how to choose their parameters
#TODO inside iterator init all interesting thing logging send to wandb