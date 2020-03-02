import os
import numpy as np
import torch
import torch.nn as nn
from edflow import TemplateIterator, get_logger

import torchvision
from conv_ae import Model
import matplotlib.pyplot as plt
from edflow.hooks.checkpoint_hooks.common import get_latest_checkpoint
#os.environ["CUDA_VISIBLE_DEVICES"] = "9"    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import imageio
from skimage import img_as_ubyte

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
            
        
        in_img = inputs.cpu().detach().permute(0,2,3,1).numpy()
        out_img = outputs.cpu().detach().permute(0,2,3,1).numpy()
        #print("in_img.shape:",in_img.shape)
        #print("in_img[0].shape:",in_img[0].shape)
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
        
        for i in range(self.config["batch_size"]):
            log_dict["images"].update({"input_" + str(i): np.expand_dims(in_img[i],0)})
            log_dict["images"].update({"output_" + str(i): np.expand_dims(out_img[i],0)})

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

def callback_latent(root, data_in, data_out, config):
    # create all needed paths
    root = root[:root.find("/eval/")]
    checkpoint_path = root + "/train/checkpoints/" 

    pca_path = root + "/pca/"
    if not os.path.isdir(pca_path):
        os.mkdir(pca_path)
    
    figures_path = root + "/pca/figures/"
    if not os.path.isdir(figures_path):
        os.makedirs(figures_path)
    print("figures will be created at:", figures_path)
    
    gif_path = figures_path + "gifs/"
    if not os.path.isdir(gif_path):
            os.makedirs(gif_path)
    
    name_start = root.rfind("/")
    run_name = root[name_start+20:]

    # calculate the eigenvetors and eigenvalues
    data_out.show()
    latents = data_out.labels["latent_rep"]
    eig_val, eig_vec = np.linalg.eig(latents.transpose()@latents)
    argsort_eig_val = np.argsort(eig_val)
    
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
    x = torch.linspace(mu-delta,mu+delta,num_images).to(device)
    pc = torch.from_numpy(eig_vec[pc_num]).to(device)

    z = torch.zeros([num_images,config["linear"]["latent_dim"]], dtype=torch.float).to(device)
    for i in range(num_images):
        z[i] = pc * x[i]
    img = model.sample_gaus(z)
    img = img.detach().cpu().permute([0,2,3,1]).numpy()
    img = (img + 1)/2
    final = np.hstack(img)
    return final

def view_pc(model, eig_vec, config, mu=0, delta=20, num_pc=1, num_images=5, save=False, figures_path=None):
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
        img = model.sample_gaus(z)
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
    xx = torch.linspace(mu-delta,mu+delta,numb_images).to(device)
    pc = torch.from_numpy(eig_vec[pc_num]).to(device).double()
    
    all_images = []
    z = torch.zeros([numb_images,config["linear"]["latent_dim"]], dtype=torch.float).to(device)
    for i in range(numb_images):
        z[i] = pc * xx[i]
    img = model.sample_gaus(z)
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
    import conv_ae
    import matplotlib.pyplot as plt
    
    vae = conv_ae.Model(config)

    #checkpoint_path = root + 
    #state = torch.load(checkpoint_path)
    self.model.load_state_dict(state["model"])
    self.optimizer.load_state_dict(state["optimizer"])
    
    amount_of_pics = 2
    interpolate = np.linspace(-3,3,amount_of_pics)

    for i in range(len(interpolate)):
        smaple = pc1 * interpolate[i]
        x = vae.sample_gaus()
        plt.imshow(x)
        plt.show
'''

#    print(sort_eig_val[-10:])

#TODO look at optimzer in config and how to choose their parameters
#TODO inside iterator init all interesting thing logging send to wandb