import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte

import torch
import torch.nn as nn
import torchvision

from edflow import TemplateIterator, get_logger
from edflow.hooks.checkpoint_hooks.common import get_latest_checkpoint

from model.vae import VAE_Model

class Iterator(TemplateIterator):
    def __init__(self, config, root, model, *args, **kwargs):
        super().__init__(config, root, model, *args, **kwargs)
        # export to the right gpu if specified in the config
        if "CUDA_VISIBLE_DEVICES" in self.config:
            if type(self.config["CUDA_VISIBLE_DEVICES"]) != str:
                self.config["CUDA_VISIBLE_DEVICES"] = str(self.config["CUDA_VISIBLE_DEVICES"])
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config["CUDA_VISIBLE_DEVICES"]
        # get the config and the logger
        self.config = config
        self.logger = get_logger("Iterator")
        # Test the config for the iterator
        self.test_config(config)
        # Log the architecture of the model
        self.logger.debug(f"{model}")
        # Move mode to device cuda if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)
        self.logger.debug(f"Model will pushed to the device: {self.device}")
        # get the loss and optimizer
        self.criterion = self.get_loss_funct(config["loss_function"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])

        if "track_loss" in self.config:
            # track an additional loss if specified
            self.track_criterion = self.get_loss_funct(self.config["track_loss"])
        if "variational" in self.config:
            # get the offset for the sigmoid regulator for the variational part of the loss 
            self.sigmoid_x_offset = self.offset_requlator()

    
    def test_config(self, config):
        '''Test the config if it will work with the iterator.''' 
        assert "loss_function" in config, "The config must contain and define a Loss function. possibilities:{'L1','L2'or'MSE','KL'or'KLD'}."
        assert "learning_rate" in config, "The config must contain and define a the learning rate."
        assert "weight_decay" in config, "The config must contain and define a the weight decay."
        
    def get_loss_funct(self, loss_function):
        '''Get the loss function specified in the config.'''
        if loss_function == "L1":
            return nn.L1Loss()
        if loss_function == "L2" or loss_function == "MSE":
            return nn.MSELoss()
        if loss_function == "KL" or loss_function == "KLD":
            return nn.KLDivLoss()

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

    def offset_requlator(self):
        '''Return the offset of the sigmoid function for the loss on mu given the parameter from the config.'''
        if "sigmoid_regulator" in self.config["variational"] and "start_step" in self.config["variational"]["sigmoid_regulator"] and "start_amplitude" in self.config["variational"]["sigmoid_regulator"] and "width" in self.config["variational"]["sigmoid_regulator"] and "amplitude" in self.config["variational"]["sigmoid_regulator"]:
            # check for all parameter
            sigmoid_x_offset = self.config["variational"]["sigmoid_regulator"]["start_step"] - self.config["variational"]["sigmoid_regulator"]["width"]/3 * (np.tan((2*self.config["variational"]["sigmoid_regulator"]["start_amplitude"])/self.config["variational"]["sigmoid_regulator"]["amplitude"] - 1))
        else:
            self.logger.info("parameters for sigmoid regulator are not specified now choosing default.")
            self.config["variational"]["sigmoid_regulator"]["start_step"] = 50000
            self.config["variational"]["sigmoid_regulator"]["start_amplitude"] = 10**(-5) 
            self.config["variational"]["sigmoid_regulator"]["width"] = 2000
            self.config["variational"]["sigmoid_regulator"]["amplitude"] = 0.001
            sigmoid_x_offset = self.offset_requlator()
        return sigmoid_x_offset

    def sigmoid_regulator(self, step):
        '''Return the sigmoid regulator for the absolute loss of the varaitional expected value mu.'''
        return self.config["variational"]["sigmoid_regulator"]["amplitude"] * (0.5 + 0.5 * np.tanh((step - self.sigmoid_x_offset)*3/self.config["variational"]["sigmoid_regulator"]["width"]))

    def step_op(self, model, **kwargs):
        '''This function will be called every step in the training.'''
        # get inputs
        inputs = kwargs["image"]
        inputs = torch.tensor(inputs).to(self.device)
        
        # compute the output of the model
        results = model(inputs)
        outputs = results
        self.logger.debug("inputs.shape: " + str(inputs.shape))
        self.logger.debug("output.shape: " + str(outputs.shape))
        
        # compute the reconstruction loss
        loss = self.criterion(outputs, inputs)
        mean_loss = torch.mean(loss)
        min_loss = np.min(loss.cpu().detach().numpy())
        max_loss = np.max(loss.cpu().detach().numpy())    
        
        if "variational" in self.config:
            # Calculate loss for latent representation if the model is a VAE
            step = torch.tensor(self.get_global_step(), dtype = torch.float)
            abs_z = torch.mean(torch.abs(model.z))
            # force absolute value of latent representation (with a regulation factor) into the loss to minimize it
            abs_loss = self.sigmoid_regulator(step).to(self.device) * abs_z
            
            if "sigma" in self.config["variational"] and self.config["variational"]["sigma"]:
                print("not ready yet")
                #TODO implement kl loss
            else:
                mean_loss += abs_loss
        # create a dictionary to log with all interesting variables 
        log_dict = {
                "images": {},
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
        # log the input and output images
        in_img = inputs.cpu().detach().permute(0,2,3,1).numpy()
        out_img = outputs.cpu().detach().permute(0,2,3,1).numpy()
        for i in range(self.config["batch_size"]):
            log_dict["images"].update({"input_" + str(i): np.expand_dims(in_img[i],0)})
            log_dict["images"].update({"output_" + str(i): np.expand_dims(out_img[i],0)})
        # calculate a secound reconstruction loss to log if wanted 
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
            # This function will be executed if the model is in training mode
            self.optimizer.zero_grad()
            mean_loss.backward()
            self.optimizer.step()
            # do not log images if in training mode
            del log_dict["images"]

        def log_op():
            # This function will always execute
            return log_dict

        def eval_op():
            # This function will be executed if the model is in evaluation mode
            return {"labels": {"latent_rep": model.z.cpu().detach().numpy()}}
    
        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

def callback_latent(root, data_in, data_out, config):
    """This function will create do a pca on the latent representation and save results in figures.
    
    :param root: The root data path point to evaluation folder in the current run.
    :type root: String
    :param data_out: The latent represenation of the model.
    :type data_out: Dictionary
    :param config: The config in which the run was initialised. 
    :type config: Dictionary
    """    
    # move model to gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create all needed paths
    root = root[:root.find("/eval/")]
    checkpoint_path = root + "/train/checkpoints/" 
    # craete the path where the pca results are going to be saved
    pca_path = root + "/pca/"
    if not os.path.isdir(pca_path):
        os.mkdir(pca_path)
    # create path to the figures
    figures_path = root + "/pca/figures/"
    if not os.path.isdir(figures_path):
        os.makedirs(figures_path)
    print("figures will be saved at:", figures_path)
    # create path to where the gifs are saved
    gif_path = figures_path + "gifs/"
    if not os.path.isdir(gif_path):
            os.makedirs(gif_path)
    # find the name of the run
    name_start = root.rfind("/")
    run_name = root[name_start+20:]

    # calculate the eigenvetors and eigenvalues
    data_out.show()
    latents = data_out.labels["latent_rep"]
    eig_val, eig_vec = np.linalg.eig(latents.transpose()@latents)
    argsort_eig_val = np.argsort(eig_val)
    # sort the eigenvectors according to their eigen values
    pc_vec_sorted = np.zeros(eig_vec.shape)
    for i in range(1,len(argsort_eig_val)):
        pc_vec_sorted[i-1] = eig_vec[argsort_eig_val[-i]]
    pc_val_sorted = np.sort(eig_val)[::-1]
    # save the eigenvectors and the eigenvalues
    np.save(pca_path + "eig_vec.npy", pc_vec_sorted)
    np.save(pca_path + "eig_val.npy", pc_val_sorted)
    eig_vec = pc_vec_sorted
    eig_val = pc_val_sorted
    
    # load Model with latest checkpoint
    vae = Model(config)
    latest_chkpt_path = get_latest_checkpoint(checkpoint_root = checkpoint_path)
    vae.load_state_dict(torch.load(latest_chkpt_path)["model"])
    vae = vae.to(device)
    
    # looking at the latent representation
    plot_eig_val(eig_val = eig_val, figures_path=figures_path, save=True)
    view_pc(model=vae, eig_vec=eig_vec, mu=0, delta=60,num_pc=10, num_images=9,save=True, figures_path = figures_path, config = config)
    save_image_seq(model = vae, config = config, eig_vec = eig_vec, gif_path = gif_path, mu = 0, delta=60, pc_num=np.arange(0,8,1), numb_images=100, loop_gif=True)
    pc_start = [0,2,4,6]
    for i in range(len(pc_start)):
        view_two_added_pc(model=vae, eig_vec=eig_vec, run_name=run_name, config = config, mu = 0, delta = 50, pc_num_start=pc_start[i], num_images_x=8, num_images_y=8, save=True, figures_path=figures_path)

def plot_eig_val(eig_val, figures_path, save):
    '''Save a plot with the eigenvalues.'''
    idx_lambda = np.arange(0,len(eig_val),1)
    plt.figure(figsize=(15,7))
    plt.bar(idx_lambda, eig_val)
    plt.ylabel("eigenvalue $\lambda$")
    plt.xlabel("index of eigenvector")
    if save:
        plt.savefig(figures_path + "eig_value_plot.jpg")
        print("Plot of eigenvalues is saved.")
    else:
        plt.show()

def get_one_pc(model, eig_vec, config, mu = 0, delta=40, pc_num=0, num_images=8):
    '''Create images for one principle component in the range mu-delta to mu+delta and return them.'''
    x = torch.linspace(mu-delta,mu+delta,num_images).to(device)
    pc = torch.from_numpy(eig_vec[pc_num]).to(device)

    z = torch.zeros([num_images,config["linear"]["latent_dim"]], dtype=torch.float).to(device)
    for i in range(num_images):
        z[i] = pc * x[i]
    img = model.latent_sample(z)
    img = img.detach().cpu().permute([0,2,3,1]).numpy()
    img = (img + 1)/2
    final = np.hstack(img)
    return final

def view_pc(model, eig_vec, config, mu=0, delta=20, num_pc=1, num_images=5, save=False, figures_path=None):
    '''Create on big plot with a given number of images for as many pcs as specified and save the plot.'''
    pc = np.arange(0,num_pc,1)
    final = []
    for i in range(num_pc):
        final.append(get_one_pc(model = model, eig_vec = eig_vec, config = config, mu = mu, delta = delta, pc_num = pc[i], num_images = num_images))
    
    final = np.asarray(final)
    final = np.vstack(final)
    plt.figure(figsize=(num_images*1.5,num_pc*1.5))
    
    plt.xlabel("factor of principle components")
    plt.ylabel("principle component")
    lab_pos_x = np.arange(32,64*num_images,64)
    x_lab = np.linspace(mu-delta,mu+delta,num_images)
    lab_pos_y = np.arange(32,64*num_pc,64)
    plt.xticks(lab_pos_x, np.round(x_lab,1))
    plt.yticks(lab_pos_y, pc)
    plt.imshow(final)
    if save:
        plt.savefig(figures_path + "view_pc__amount_of_pc" + str(num_pc) + "_num_images" + str(num_images) + "_mu" + str(mu) + "delta" + str(delta) + ".jpg" )
        print("Plot of PC's is saved.")
    else:
        plt.show()
    
def view_two_added_pc(model, eig_vec, run_name, config, mu = 0, delta = 20, pc_num_start=0, num_images_x=5, num_images_y=5, save=False, figures_path=None):
    '''Create a big plot with two pcs along one axis each which are added in between and save the plot.'''
    if type(mu) not in [list,np.ndarray]:
        mu = [mu, mu]
        delta = [delta, delta]
    x1 = torch.linspace(mu[0]-delta[0],mu[0]+delta[0], num_images_x)
    x2 = torch.linspace(mu[1]-delta[1],mu[1]+delta[1], num_images_y)
    
    pc1 = torch.from_numpy(eig_vec[pc_num_start]).to(device)
    pc2 = torch.from_numpy(eig_vec[pc_num_start+1]).to(device)
    
    z = torch.zeros([num_images_x,config["linear"]["latent_dim"]], dtype=torch.float).to(device)
    h_final = []
    for j in range(num_images_y):
        for i in range(num_images_x):
            z[i] = pc1 * x1[i] + pc2 * x2[j]
        img = model.latent_sample(z)
        img = img.detach().cpu().permute([0,2,3,1]).numpy()
        img = (img + 1)/2
        h_final.append(np.hstack(img))
    h_final = np.asarray(h_final)
    final = np.vstack(h_final)
    plt.figure(figsize=(num_images_x*1.5,num_images_y*1.5))
    
    plt.title("Two added PC's from: " + run_name)
    plt.imshow(final)
    plt.xlabel("factor of principle component " + str(pc_num_start))
    plt.ylabel("factor of principle component " + str(pc_num_start+1))
    lab_x_pos = np.arange(32,64*num_images_x,64)
    lab_y_pos = np.arange(32,64*num_images_y,64)
    x_lab = x1.detach().cpu().numpy()
    y_lab = x2.detach().cpu().numpy()
    plt.xticks(lab_x_pos, np.round(x_lab,1))
    plt.yticks(lab_y_pos, np.round(y_lab,1))
    if save:
        plt.savefig(figures_path + "2added_pc__start_pc_" + str(pc_num_start) + "_mu_1_" + str(mu[0]) + "__mu_2_" + str(mu[1]) + ".jpg" )
        print("Plot of two added PC's is saved.")
    else:
        plt.show()

# functions to create gifs
def save_image_seq(model, config, eig_vec, gif_path, mu = 0, delta=50, pc_num=0, numb_images=250, loop_gif=False):
    '''Create a gif in which any amount of pcs change in their value in a gifen range and save it.'''
    if type(pc_num) in [list,np.ndarray]:
        gifs = []
        for i in range(len(pc_num)):
            all_images = create_image_seq(model = model, config = config, eig_vec = eig_vec, mu = mu, delta = delta, pc_num = pc_num[i], numb_images = numb_images, loop_gif = loop_gif)
            gifs.append(all_images)
        gifs = np.asarray(gifs)
        gifs = np.concatenate((gifs[:]),axis=2)
        imageio.mimsave(gif_path + "pc_numbers" + str(pc_num[0]) + "-" + str(pc_num[-1]) + "_mu" + str(mu) + "_delta" + str(delta) + '.gif', gifs)
    else:
        all_images = create_image_seq(mu, delta, pc_num, numb_images)
        imageio.mimsave(gif_path + "pc_numb" + str(pc_num) + "_mu" + str(mu) + "_delta" + str(delta) + '.gif', all_images)
    print("Gif of variing PC's is saved.")

def create_image_seq(model, config, eig_vec, mu = 0, delta=50, pc_num=0, numb_images=250, loop_gif=False):
    '''Create a given amount of images for a pc in a certain range and return the images'''
    xx = torch.linspace(mu-delta,mu+delta,numb_images).to(device)
    pc = torch.from_numpy(eig_vec[pc_num]).to(device).double()
    
    all_images = []
    z = torch.zeros([numb_images,config["linear"]["latent_dim"]], dtype=torch.float).to(device)
    for i in range(numb_images):
        z[i] = pc * xx[i]
    img = model.latent_sample(z)
    img = img.detach().cpu().permute([0,2,3,1]).numpy()
    img = (img + 1)/2
    img = img_as_ubyte(img)
    for i in range(numb_images):
        all_images.append(img[i])
    if loop_gif:
        for i in range(numb_images):
            all_images.append(img[-i-1])
    return all_images














'''
    state = torch.load(checkpoint_path)
    self.model.load_state_dict(state["model"])
    self.optimizer.load_state_dict(state["optimizer"])


    pc1 = eig_vec[argsort_eig_val[-1]]
    pc2 = eig_vec[argsort_eig_val[-2]]
    sort_eig_val = np.sort(eig_val)
    latest_chkpt_path = edflow.edflow.hooks.checkpoint_hooks.common.get_latest_checkpoint(checkpoint_root = checkpoint_path)

    #from edflow import 
    import vae
    import matplotlib.pyplot as plt
    
    vae = vae.Model(config)

    #checkpoint_path = root + 
    #state = torch.load(checkpoint_path)
    self.model.load_state_dict(state["model"])
    self.optimizer.load_state_dict(state["optimizer"])
    
    amount_of_pics = 2
    interpolate = np.linspace(-3,3,amount_of_pics)

    for i in range(len(interpolate)):
        smaple = pc1 * interpolate[i]
        x = vae.latent_sample()
        plt.imshow(x)
        plt.show
'''

#    print(sort_eig_val[-10:])

#TODO look at optimzer in config and how to choose their parameters
#TODO inside iterator init all interesting thing logging send to wandb