import os
import numpy as np
import torch
import torch.nn as nn
from edflow import TemplateIterator, get_logger

class Iterator(TemplateIterator):
    def __init__(self, config, root, model, *args, **kwargs):
        super().__init__(config, root, model, *args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # set the right gpu specified in config
        if "CUDA_VISIBLE_DEVICES" in self.config:
            if type(self.config["CUDA_VISIBLE_DEVICES"]) != str:
                self.config["CUDA_VISIBLE_DEVICES"] = str(self.config["CUDA_VISIBLE_DEVICES"])
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "5"
        
        self.config = config
        self.logger = get_logger("Iterator")
        self.check_config(config)
        self.logger.debug(f"{model}")
        self.logger.debug(f"{self.device}")

        model = model.to(self.device)
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
        inputs = torch.tensor(inputs).to(self.device)
        
        # compute loss
        results = model(inputs)
        outputs = results

        self.logger.debug("inputs.shape: " + str(inputs.shape))
        self.logger.debug("output.shape: " + str(outputs.shape))
        
        loss = self.criterion(outputs, inputs)
        mean_loss = torch.mean(loss)
        #print("mean_loss:",mean_loss)
        min_loss = np.min(loss.cpu().detach().numpy())
        max_loss = np.max(loss.cpu().detach().numpy())    
        if "variational" in self.config:
            step = torch.tensor(self.get_global_step(), dtype = torch.float)
            abs_z = torch.mean(torch.abs(model.z))
            if "tanh" in self.config["variational"]:
                    if "mean" in self.config["variational"]["tanh"] and "width" in self.config["variational"]["tanh"] and "factor" in self.config["variational"]["tanh"]:
                        regulator = self.config["variational"]["tanh"]["factor"] * (0.5 + 0.5 * nn.functional.tanh((step - self.config["variational"]["tanh"]["mean"])/self.config["variational"]["tanh"]["width"]))
                    else:
                        regulator = 0.5 * (0.5 + 0.5 * nn.functional.tanh((step-1000)/150))
            regulator = regulator.to(self.device)
            abs_loss = regulator * abs_z
            #print("abs_z:", abs_z)
            #print("regulator:", regulator)
            #print("abs_loss:", abs_loss)
            #print("mean_loss:", mean_loss)
            if "sigma" in self.config["variational"] and self.config["variational"]["sigma"]:
                print("not ready yet")
                # implement kl loss
            else:
                mean_loss += abs_loss

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

        if "track_loss" in self.config:
            track_loss = self.track_criterion(outputs, inputs)
            track_mean_loss = torch.mean(track_loss)
            track_min_loss = np.min(track_loss.cpu().detach().numpy())
            track_max_loss = np.max(track_loss.cpu().detach().numpy())    
            log_dict["scalars"].update({
                self.config["track_loss"]:
                {
                    "min_loss": track_min_loss,
                    "max_loss": track_max_loss,
                    "mean_loss": track_mean_loss
                }
            })
            
        def train_op():
            self.optimizer.zero_grad()
            mean_loss.backward()
            self.optimizer.step()
            del log_dict["images"]

        def log_op():
            return log_dict

        def eval_op():
            return {"labels": {"latent_rep": model.z.cpu().detach().numpy()}}
        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}


def call_back_test(root, data_in, data_out, config):
    data_out.show()
    latents = data_out.labels[lstennnn]

#TODO look at optimzer in config and how to choose their parameters
#TODO inside iterator init all interesting thing logging send to wandb