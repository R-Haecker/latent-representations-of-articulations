import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte

from edflow.hooks.checkpoint_hooks.common import get_latest_checkpoint

from torchfid import FIDScore

import torch
import torch.nn as nn
import sys
sys.path.append('../')
from model.gan import GAN
#VAE_Model


def latent(root, data_in, data_out, config):
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
    first_run = None
    if not os.path.isdir(figures_path):
        os.makedirs(figures_path)
        first_run = True
    else:
        first_run = False
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
    gan = GAN(config)
    latest_chkpt_path = get_latest_checkpoint(checkpoint_root = checkpoint_path)
    vae = gan.generator
    vae.load_state_dict(torch.load(latest_chkpt_path)["netG"])
    vae = vae.to(device)
    
    # looking at the latent representation
    if "latent_delta" in config:
        delta_ = config["latent_delta"]
        print("delta is set to ", delta_)
    else:
        delta_ = 10
    if first_run:
        plot_eig_val(eig_val = eig_val, figures_path=figures_path, save=True)
    if "linear" in config:
        num_pc = config["linear"]["latent_dim"]
    else:
        num_pc = 9
    view_pc(model=vae, eig_vec=eig_vec, mu=0, delta=delta_,num_pc=num_pc, num_images=9,save=True, figures_path = figures_path, config = config)
    save_image_seq(model = vae, config = config, eig_vec = eig_vec, gif_path = gif_path, mu = 0, delta=delta_, pc_num=num_pc, numb_images=100, loop_gif=True)
    if num_pc > 1:
        cur_num = 0
        pc_start = []
        for i in range(num_pc//2):
            pc_start.append(cur_num)
            cur_num += 2
        for i in range(len(pc_start)):
            view_two_added_pc(model=vae, eig_vec=eig_vec, run_name=run_name, config = config, mu = 0, delta = delta_, pc_num_start=pc_start[i], num_images_x=8, num_images_y=8, save=True, figures_path=figures_path)

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
    # move model to gpu if possible
    device = "cuda" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.linspace(mu-delta,mu+delta,num_images).to(device)
    pc = torch.from_numpy(eig_vec[pc_num]).to(device)
    if "linear" in config:
        latent_dim = config["linear"]["latent_dim"]
    else:
        latent_dim = config["conv"]["n_channel_max"]
    z = torch.zeros([num_images, latent_dim], dtype=torch.float).to(device)
    for i in range(num_images):
        z[i] = pc * x[i]
    z = z.to(device)
    img = model.latent_sample(z, batch_size = num_images)
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
    # move model to gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if type(mu) not in [list,np.ndarray]:
        mu = [mu, mu]
        delta = [delta, delta]
    x1 = torch.linspace(mu[0]-delta[0],mu[0]+delta[0], num_images_x)
    x2 = torch.linspace(mu[1]-delta[1],mu[1]+delta[1], num_images_y)
    print("pcnum_start",pc_num_start)
    pc1 = torch.from_numpy(eig_vec[pc_num_start]).to(device)
    pc2 = torch.from_numpy(eig_vec[pc_num_start+1]).to(device)
    
    if "linear" in config:
        latent_dim = config["linear"]["latent_dim"]
    else:
        latent_dim = config["conv"]["n_channel_max"]
    z = torch.zeros([num_images_x, latent_dim], dtype=torch.float).to(device)
    h_final = []
    for j in range(num_images_y):
        for i in range(num_images_x):
            z[i] = pc1 * x1[i] + pc2 * x2[j]
        img = model.latent_sample(z, batch_size = num_images_x)
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
        plt.savefig(figures_path + "2added_pc_delta_" + str(delta) + "_start_pc_" + str(pc_num_start) + "_mu_1_" + str(mu[0]) + "__mu_2_" + str(mu[1]) + ".jpg" )
        print("Plot of two added PC's is saved.")
    else:
        plt.show()

# functions to create gifs
def save_image_seq(model, config, eig_vec, gif_path, mu = 0, delta=50, pc_num=0, numb_images=250, loop_gif=False):
    '''Create a gif in which any amount of pcs change in their value in a gifen range and save it.'''
    pc = np.arange(0,pc_num,1)
    gifs = []
    for i in range(pc_num):
        all_images = create_image_seq(model = model, config = config, eig_vec = eig_vec, mu = mu, delta = delta, pc_num = pc[i], numb_images = numb_images, loop_gif = loop_gif)
        gifs.append(all_images)
    gifs = np.asarray(gifs)
    gifs = np.concatenate((gifs[:]),axis=2)
    imageio.mimsave(gif_path + "pc_numbers_0-" + str(pc_num) + "_mu" + str(mu) + "_delta" + str(delta) + '.gif', gifs)
    print("Gif of variing PC's is saved.")

def create_image_seq(model, config, eig_vec, mu = 0, delta=50, pc_num=0, numb_images=250, loop_gif=False):
    '''Create a given amount of images for a pc in a certain range and return the images'''
    # move model to gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xx = torch.linspace(mu-delta,mu+delta,numb_images).to(device)
    pc = torch.from_numpy(eig_vec[pc_num]).to(device).double()
    
    all_images = []
    if "linear" in config:
        latent_dim = config["linear"]["latent_dim"]
    else:
        latent_dim = config["conv"]["n_channel_max"]
    z = torch.zeros([numb_images, latent_dim], dtype=torch.float).to(device)
    for i in range(numb_images):
        z[i] = pc * xx[i]
    img = model.latent_sample(z, batch_size = numb_images)
    img = img.detach().cpu().permute([0,2,3,1]).numpy()
    img = (img + 1)/2
    img = img_as_ubyte(img)
    for i in range(numb_images):
        all_images.append(img[i])
    if loop_gif:
        for i in range(numb_images):
            all_images.append(img[-i-1])
    return all_images

# TODO implement sigma sampling

def latent_save_images(root, data_in, data_out, config):
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
    image_path = root + "/FID/data/"
    if not os.path.isdir(image_path):
        os.mkdir(image_path)
    
    # find the name of the run
    name_start = root.rfind("/")
    full_run_name = root[name_start:]
    run_name = root[name_start+20:]


    # calculate the eigenvetors and eigenvalues
    #data_out.show()
    
    tramsform = torchvision.transforms.Compose([transforms.Resize(size=(64,64)), transforms.ToTensor(), 
                                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    def load_image(idx, root):
        image_path = os.path.join(root, "images/image_index_" + str(idx) + ".png")
        image = Image.fromarray(io.imread(image_path)) 
        image = tramsform(image)
        return image

    gan = GAN(config)
    latest_chkpt_path = get_latest_checkpoint(checkpoint_root = checkpoint_path)
    vae = gan.generator
    vae.load_state_dict(torch.load(latest_chkpt_path)["netG"])
    vae = vae.to(device)

    indices = data_out.labels["index"]
    for i in range(indices.shape[0]):
        print("progress",100*i/indices.shpe[0])
        idx = indices[i]
        img = load_image(idx, config["data_root"])
        output_image = vae(img)
        img.detache().cpu().numpy()
        save_image(img, image_path + "input/" + 'image_' + str(i) + '.png')
        output_image.detache().cpu().numpy()
        save_image(output_image, image_path + "output/" + 'image_' + str(i) + '.png')
    
    print("done with saving")

    fid_score = FIDScore(batch_size=50, verbose=True)
    score = fid_score(image_path + "input/", image_path + "output/")
    
    np.save(image_path + "fid_score.npy" ,score)

    file1 = open(image_path + "fid_score.txt","w") 
    L = ["FID score of run" + full_run_name " = " + str(score)]  
    file1.write(L[0]) 
    file1.close()
    new_path = "/export/home/rhaecker/documents/research-of-latent-representation/VAE/research/FID/"
    np.save(image_path + full_run_name + "fid_score.npy" ,score)

    file1 = open(image_path+ full_run_name + "fid_score.txt","w") 
    L = ["FID score of run" + full_run_name " = " + str(score)]  
    file1.write(L[0]) 
    file1.close()