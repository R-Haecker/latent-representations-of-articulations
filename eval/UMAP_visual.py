import yaml
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

import imageio
from skimage import img_as_ubyte
from PIL import Image

sys.path.append('/export/home/rhaecker/documents/research-of-latent-representation/GAN')
from model.gan import GAN, VAE_Model

import time
import json
import umap
import numpy as np
import matplotlib.pyplot as plt
from edflow.hooks.checkpoint_hooks.common import get_latest_checkpoint
from edflow.data.believers.meta import MetaDataset
from skimage import io
#os.environ["CUDA_VISIBLE_DEVICES"] = "9"    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UMAP_():
    def __init__(self, run_name):
        self.run_name = run_name
        self.root, self.config = self.init_needed_parameters(need_data_out=False)
        self.results_path = self.get_run_results_path()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.init_vae()
        
        # get all phi loaded
        self.amount_phi=50
        self.delta_phi=360
        self.amount_theta=50
        self.delta_theta=180
        self.phi_offset=0
        #path_image_seq_dataset = self.get_path_seq_dataset(amount_phi, delta_phi)
        images = self.load_torch_images_big()
        
        #img = images.detach().cpu().numpy()
        #print("img.shape",img.shape)
        #self.plot_t_images(img,path=self.results_path)
        '''l_dict = []
        for i in range(img.shape[0]):
            image = img[i]+0.5
            dict_ = {}
            dict_["image"] = image
            l_dict.append(dict_)
        '''
            
        data_latent_torch = self.model.encode_images_to_z(images)
        self.all_latent_data = data_latent_torch.detach().cpu().numpy()
        print("after all latent_data")

    def plot_t_images(self, images, save =False, name="", path=None):
        def get_row_images(images, row_lemgth=8):
            images = images/2 + 0.5
            final = np.zeros([64,512,3])
            for i in range(8):
                cur_img = np.rot90(np.rot90(np.rot90(np.reshape(np.transpose(images[i]),(64,64,3)))))
                final[:,i*64:i*64+64,:] = cur_img
            return final
        
        plt.figure(figsize=(8,8))
        if type(images)==list:
            row_list = []
            for i in range(len(images)):
                cur_row = get_row_images(images=images[i])
                row_list.append(cur_row)
            final = np.vstack(row_list)
            plt.vlines(range(0,64*8,64),0,64*2)
            plt.hlines(range(64,len(images)*64,64),0,64*8)
            plt.yticks((32,64+32),("input","output"))
            phi = self.get_phi_parameter()
            phi = np.around(phi,1)
            plt.xticks(range(32,64*8+32,64),phi)
            plt.xlabel("articulation parameter $\Phi$")
        else:
            print("images.shape",images.shape)
            final = get_row_images(images=images)
            plt.vlines(range(0,64*8,64),0,64)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(name)
            
        print("final.shape",final.shape)
        plt.imshow(final)
        if save:
            plt.savefig(path + "images_plot_" + name + "_delta_phi_" + str(self.delta_phi) + "_offset_" + str(self.phi_offset) + ".png",dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_images(self, dicts, images_per_row = 8, save_fig = False, path = ""):
        """Plot and show all images contained in the list dicts of dictionaries and label them with their corresponding index."""
        # How many images are there
        numb = len(dicts)

        if numb < images_per_row:
            images_per_row = numb
        numb_y = 0
        numb_x = images_per_row
        # numb_y and numb_x determine the size of the grid of images
        while numb > numb_x:
            numb_y += 1
            numb -= images_per_row
        numb_y += 1

        print("numb_y: " + str(numb_y))
        print("dicts[0]['image'].shape[0]: " + str(dicts[0]["image"].shape[0]))
        print("(images_per_row*dicts[0][image].shape[1]: " + str((images_per_row*dicts[0]["image"].shape[1])))
        print("dicts[0][image].shape[2]: " + str(dicts[0]["image"].shape[2]))
        h_stacked = np.ones((numb_y, dicts[0]["image"].shape[0], (images_per_row*dicts[0]["image"].shape[1]), dicts[0]["image"].shape[2]), dtype=int)
        print("h_stacked.shape: " + str(h_stacked.shape))
        for i in range(numb_y):
            img_hstack = np.ones((numb_x, dicts[0]["image"].shape[0], dicts[0]["image"].shape[1], dicts[0]["image"].shape[2]), dtype=int)
            if i == numb_y-1:
                for j in range(numb):
                    img_hstack[j] = dicts[i*numb_x+j]["image"]
            else:    
                for j in range(numb_x):
                    img_hstack[j] = dicts[i*numb_x+j]["image"]
            h_stacked[i] = np.hstack((img_hstack))
        final_img = np.flip(np.vstack((h_stacked)),0)
        inn = Image.fromarray(final_img, "RGB")
        # Stack all rows vertically together
        #matplotlib.use('TkAgg')
        print("final_img.shape: " + str(final_img.shape))
        #final_img = np.reshape(final_img, (64,512,3))
        #print("final_img.shape: " + str(final_img.shape))
        #plt.figure(figsize=(9,2))
        plt.imshow(inn)
        #plt.axis('off')
        #plt.xlabel("latent dimension")
        #plt.yticks([])
        #k_s=["3","3","4","4","5","5","6","6"]
        #stride = [1,2,4,8,16,32,64,128,256]
        #ticks = []
        #for i in range(9):
            #ticks.append("kernel \n size " + k_s[i] + ",\n stride " + str(stride[i]))
        #    ticks.append( str(stride[i]))
        #plt.xticks(range(32-10,22+44*9,44), ticks)
        # Plot lines between images to better see the borders of the images
        #for i in range(1,9):
            #plt.axhline(y = dicts[0]["image"].shape[1]*i-0.7, color="k")
        #for i in range(1,9):
        #    plt.axvline(x = dicts[0]["image"].shape[1]*i-1, color="k")
        # Plot the index of the images onto the images
        #plt.xlim(0, 8*dicts[0]["image"].shape[1]-1)
        #plt.ylim(0, -(1)*dicts[0]["image"].shape[0]-1)
        # Save figure if wanted.
        if save_fig:
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + "/final_latent_dim.png", bbox_inches='tight', dpi=300)
        plt.show()

        
    def init_needed_parameters(self, need_data_out=True):
        # initialise root, data_out and config
        def load_config(run_path):
            run_config_path = run_path + "/configs/"
            run_config_path = run_config_path + os.listdir(run_config_path)[-1]
            with open(run_config_path) as fh:
                config = yaml.full_load(fh)
            return config

        def load_data_mem_map(run_path):
            meta_path, raw_data_path = get_raw_data_path(run_path=run_path)
            Jdata = MetaDataset(meta_path)
            return Jdata
        run_name = self.run_name
        assert run_name[0] != "/"
        prefix = "GAN/logs/"
        run_path = prefix + self.run_name
        working_directory = '/export/home/rhaecker/documents/research-of-latent-representation/'
        if run_path[-1]==["/"]:
            run_path = run_path[:-1]
        root = working_directory + run_path + "/"

        config = load_config(root)
        if need_data_out:
            data_out = load_data_mem_map(root)
            return root, data_out, config
        else:
            return root, config
        
    def get_run_results_path(self):
        a = time.time()
        timeObj = time.localtime(a)
        cur_time ='%d-%d-%d_%d-%d-%d' % (
        timeObj.tm_year, timeObj.tm_mon, timeObj.tm_mday, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec)
        working_res_directory = '/export/home/rhaecker/documents/research-of-latent-representation/VAE/research/notebooks/new_results/interpol_phi_lin_spher/'
        name_start = self.run_name.rfind("/")
        r_name = self.run_name[name_start+20:]
        results_path = working_res_directory + "/results_" + cur_time + "_" + r_name + "/"
        if not os.path.isdir(results_path):
            os.makedirs(results_path)
        return results_path
    
    def init_vae(self):
        # create all needed paths
        root = self.root
        if "/eval/" in root:
            root = root[:root.find("/eval/")]
        if root[-1] == "/":
            root = root[:-1]
        checkpoint_path = root + "/train/checkpoints/" 
        # find the name of the run
        name_start = root.rfind("/")
        run_name = root[name_start+20:]    
        # load Model with latest checkpoint
        gan = GAN(self.config)
        latest_chkpt_path = get_latest_checkpoint(checkpoint_root = checkpoint_path)
        print("latest_chkpt_path",latest_chkpt_path)
        vae = gan.generator
        vae.load_state_dict(torch.load(latest_chkpt_path)["netG"])
        vae = vae.to(self.device)
        return vae
    
    def get_path_seq_dataset(self):#, amount_phi, delta_phi):
        #path_image_seq_dataset = "/export/home/rhaecker/documents/research-of-latent-representation/data/umap_sequences/"
        #phi_range = [0,delta_phi]
        #path_image_seq_dataset = path_image_seq_dataset + str(amount_phi) + "_" + str(phi_range[0]) + "_to_" + str(phi_range[1]) + "/"
        path_image_seq_dataset = "/export/home/rhaecker/documents/research-of-latent-representation/data/phi_seq/"
        return path_image_seq_dataset
    
    def get_path_big_dataset(self):
        path_image_seq_dataset = "/export/home/rhaecker/documents/research-of-latent-representation/data/umap_sequences/cuboids_phi_theta/2_cuboids_var_phi_theta/"
        return path_image_seq_dataset
    
    
    def get_phi_parameter(self):
        data_root = self.get_path_seq_dataset()
        indices = self.indices
        if type(indices) == int:
            indices = np.linspace(0,indices-1, indices, dtype=int)
        phi_list=[]
        for i in range(indices.shape[0]):
            cur_para = self.load_parameters(indices[i], data_root)
            phi_list.append(cur_para["phi"])
        phi = np.asarray(phi_list)
        print("Loaded parameter phi from data set.")
        print("phi.shape:",phi.shape," min phi:",np.min(phi)," max phi:",np.max(phi))
        return phi

    def load_parameters(self,idx, data_root):
        # load a json file with all parameters which define the image 
        parameter_path = os.path.join(data_root, "parameters/parameters_index_" + str(idx) + ".json")
        with open(parameter_path) as f:
            parameters = json.load(f)
        return parameters

    def load_torch_images_big(self):
        def load_image(idx, root):
            image_path = os.path.join(root, "images/image_index_" + str(idx) + ".png")
            image = Image.fromarray(io.imread(image_path)) 
            image = tramsform(image)
            return image
        tramsform = torchvision.transforms.Compose([transforms.Resize(size=(64,64)), transforms.ToTensor(), 
                                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        data_root = self.get_path_big_dataset()
        def get_indices():
            all_ = 9999
            theta_perc = self.delta_theta/180*99
            phi_perc = self.delta_phi/360*99
            #assert self.amount_theta<theta_perc
            #assert self.amount_phi<phi_perc
            
            phis_ind = np.linspace(0,phi_perc*100,self.amount_phi,dtype=int)
            all_indices = []
            for idx in phis_ind:
                all_indices.append(np.linspace(idx,theta_perc+idx,self.amount_theta,dtype=int))
            
            indices = np.asarray(all_indices)
            indices = indices.flatten() 
            return indices
        
        indices = get_indices()
        self.indices = indices
        all_images = torch.zeros([self.amount_phi*self.amount_theta,3,64,64]).to(self.device)
        print("data_root",data_root)
        print("indices.shape",indices.shape)
        for i in range(indices.shape[0]):
            idx = indices[i]
            all_images[i] = load_image(idx, data_root)
        return all_images
        

    
    def load_torch_images(self):
        def load_image(idx, root):
            image_path = os.path.join(root, "images/image_index_" + str(idx) + ".png")
            image = Image.fromarray(io.imread(image_path)) 
            image = tramsform(image)
            return image

        tramsform = torchvision.transforms.Compose([transforms.Resize(size=(64,64)), transforms.ToTensor(), 
                                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        data_root = self.get_path_seq_dataset()
        def get_indices():
            all_ = 9999
            perc = self.delta_phi/360
            interest_in = all_ * perc
            offset = self.phi_offset/360 * all_
            indices = np.linspace(offset,interest_in+offset,self.amount_phi,dtype=int)
            return indices
        
        indices = get_indices()
        self.indices = indices
        all_images = torch.zeros([self.amount_phi,3,64,64]).to(self.device)
        for i in range(indices.shape[0]):
            idx = indices[i]
            
            all_images[i] = load_image(idx, data_root)
        return all_images
    
    def fit_umap_for_interpolate(self, 
                                 data_latent, 
                                 interpol_lin, 
                                 interpol_sph, 
                                 fit_umap_on, 
                                 test_difference=True,
                                 n_neighbors=15,
                                 min_dist=0.1,
                                ):
        def UMAP_fit(data, 
                     n_neighbors=15,
                     min_dist=0.1,
                     ):
            reducer = umap.UMAP(n_neighbors=n_neighbors,
                                min_dist=min_dist,
                                random_state=1
                                )
            reducer.fit(data)
            return reducer
        assert fit_umap_on in ["all_phi","phi_seq","interpol_phi_seq","interpol_all_phi"]
        if fit_umap_on == "phi_seq":
            fit_data = data_latent
        elif fit_umap_on == "interpol_phi_seq":
            interpols = np.concatenate((interpol_lin, interpol_sph))
            fit_data = np.concatenate((interpols, data_latent))
        elif fit_umap_on in ["all_phi","interpol_all_phi"]:
            if fit_umap_on == "interpol_all_phi":
                interpols = np.concatenate((interpol_lin, interpol_sph))
                fit_data = np.concatenate((interpols, self.all_latent_data))
            else:
                fit_data = self.all_latent_data
            
        reducer = UMAP_fit(fit_data ,n_neighbors=n_neighbors, min_dist=min_dist)

        data_latent_red  = reducer.transform(data_latent)    
        interpol_lin_red = reducer.transform(interpol_lin)
        interpol_sph_red = reducer.transform(interpol_sph)

        if test_difference:
            print("difference interpol_lin[0] and data:latent[0] :", test_difference_arrays(interpol_lin[0],data_latent[0]))
            print("difference interpol_lin[-1] and data:latent[-1] :",test_difference_arrays(interpol_lin[-1],data_latent[-1]))
            print("difference interpol_sph[0] and data:latent[0] :", test_difference_arrays(interpol_sph[0],data_latent[0]))
            print("difference interpol_sph[-1] and data:latent[-1] :",test_difference_arrays(interpol_sph[-1],data_latent[-1]))
            print("")
            print("test_difference_arrays(reducer.transform(data_latent),reducer.transform(data_latent) :", test_difference_arrays(reducer.transform(data_latent),reducer.transform(data_latent)))
            print("test_difference_arrays(reducer.transform(interpol_lin),reducer.transform(interpol_lin)) :",test_difference_arrays(reducer.transform(interpol_lin),reducer.transform(interpol_lin)))
            print("")
            print("difference interpol_lin_red[0] and data_latent_red[0] :", test_difference_arrays(interpol_lin_red[0],data_latent_red[0]))
            print("difference interpol_lin[-1] and data_latent_red[-1] :",test_difference_arrays(interpol_lin_red[-1],data_latent_red[-1]))
            print("difference interpol_sph_red[0] and data_latent_red[0] :", test_difference_arrays(interpol_sph_red[0],data_latent_red[0]))
            print("difference interpol_sph_red[-1] and data_latent_red[-1] :",test_difference_arrays(interpol_sph_red[-1],data_latent_red[-1]))

        return data_latent_red, interpol_lin_red, interpol_sph_red
    
    def create_interpol_images(self, amount_phi = 100, delta_phi = 15, phi_offset=10, save = False, plot = True):
        self.amount_phi = amount_phi
        self.delta_phi = delta_phi
        self.phi_offset = phi_offset
        
        images = self.load_torch_images()
        
        rec_im = self.model(images)
        data_latent_torch = self.model.encode_images_to_z(images)
        
        data_latent = data_latent_torch.detach().cpu().numpy()
        
        interpol_lin = linear_interpolation(self.all_latent_data[0], self.all_latent_data[-1], amount)
        interpol_sph = spherical_interpolation(self.all_latent_data[0], self.all_latent_data[-1], amount)
        
        interpol_lin = torch.from_numpy(interpol_lin).float().to(self.device)
        interpol_sph = torch.from_numpy(interpol_sph).float().to(self.device)
        print("creating images from interpol")
        interpol_lin_images = self.model.direct_z_sample(interpol_lin)
        interpol_sph_images = self.model.direct_z_sample(interpol_sph)
        data_imges = self.model.direct_z_sample(data_latent_torch)
        
        print("done with creating images from interpol")
        
        lin_img = interpol_lin_images.detach().cpu().numpy()
        sph_img = interpol_sph_images.detach().cpu().numpy()
        input_img = images.detach().cpu().numpy()
        recon_im = rec_im.detach().cpu().numpy()
        
        #lin_img = lin_img/2 +0.5
        #sph_img = sph_img/2 +0.5
        
        #print("min",np.min(lin_img),"max",np.max(lin_img))
        if plot:
            self.plot_t_images(lin_img, name="linear interpolation",save=save, path=self.results_path,)
            self.plot_t_images(sph_img, name="spherical interpolation",save=save, path=self.results_path)
            self.plot_t_images([input_img,recon_im,lin_img,sph_img],save=save, path=self.results_path, name="input_output_image")
        
        return interpol_lin.detach().cpu().numpy(), interpol_sph.detach().cpu().numpy()
    
    
    def create_data_latent_interpolations(self, amount_phi=1000, delta_phi=180, phi_offset=0):
        self.amount_phi = amount_phi
        self.delta_phi = delta_phi
        self.phi_offset = phi_offset
        
        images = self.load_torch_images()

        data_latent_torch = self.model.encode_images_to_z(images)
        data_latent = data_latent_torch.detach().cpu().numpy()

        interpol_lin = linear_interpolation(data_latent[0], data_latent[-1], amount_phi)
        interpol_sph = spherical_interpolation(data_latent[0], data_latent[-1], amount_phi)
        
        phi = self.get_phi_parameter()
        
        self.calculate_differences_of_interpolations(data_latent, interpol_lin, interpol_sph)
        return data_latent, interpol_lin, interpol_sph, phi
    

    def calculate_differences_of_interpolations(self, data_latent, interpol_lin, interpol_sph):
        mse_lin_interpol = mse_difference_of_arrays(data_latent, interpol_lin)    
        mse_sph_interpol = mse_difference_of_arrays(data_latent, interpol_sph) 
        mse_lin_to_sph = mse_difference_of_arrays(interpol_lin, interpol_sph) 
        self.mse_lin_interpol = mse_lin_interpol
        self.mse_sph_interpol = mse_sph_interpol
        self.mse_lin_to_sph = mse_lin_to_sph
        print("MSE of linear interpolation and phi sequence:", mse_lin_interpol)
        print("MSE of spherical interpolation and phi sequence:", mse_sph_interpol)
        print("MSE between spherical and linear interpolation:", mse_lin_to_sph)

    def create_interpol_list_array_and_cmap(self, amount_phi=1000, delta_phi=180, fit_umap_on="phi_seq"):
        data_latent, interpol_lin, interpol_sph, phi = self.create_data_latent_interpolations(amount_phi, delta_phi)
        
        self.calculate_differences_of_interpolations(data_latent, interpol_lin, interpol_sph)
        
        list_scatter_array, list_scatter_cmap = self.umap_fit_and_interpol(data_latent, interpol_lin, interpol_sph, phi, fit_umap_on)
        return list_scatter_array, list_scatter_cmap
    
    def widget_fit_and_plot(self, data_latent, interpol_lin, interpol_sph,
                            fit_umap_on, save, n_neighbors=15,min_dist=0.1, start_end_point = 1,
                            cmap_list    = ['autumn',"summer","winter"],
                            marker_list  = [".","^","v"],
                            size_list    = [5,5,5],
                            z_order_list = [3,2,1],
                            alpha_list   = [1,0.5,0.5]
                            ):
        phi = self.get_phi_parameter()
        list_scatter_array, list_scatter_cmap = self.umap_fit_and_interpol(data_latent, interpol_lin, interpol_sph, phi, fit_umap_on=fit_umap_on, n_neighbors=n_neighbors, min_dist=min_dist)
        list_scatter_array[0] = self.all_latent_data
        self.plot_UMAP(list_scatter_array, [list_scatter_cmap[0]], list_label=['Parameter $\Phi$',"Spherical interpolation", "Linear interpolation"], save=save, start_end_point = start_end_point,
                        cmap_list=cmap_list,
                        marker_list=marker_list,
                        size_list=size_list,
                        z_order_list=z_order_list,
                        alpha_list=alpha_list,
                        )
    
    def umap_fit_and_interpol(
        self, 
        data_latent, 
        interpol_lin, 
        interpol_sph, 
        phi, 
        fit_umap_on="phi_seq",
        n_neighbors=15,
        min_dist=0.1
        ):
        data_latent_red, interpol_lin_red, interpol_sph_red = self.fit_umap_for_interpolate(
            data_latent, 
            interpol_lin, 
            interpol_sph, 
            fit_umap_on,
            test_difference=False,
            n_neighbors=n_neighbors,
            min_dist=min_dist
            )
        list_scatter_array = [ data_latent_red, interpol_sph_red, interpol_lin_red ]
        
        list_scatter_cmap  = [ phi, interpol_sph, interpol_lin ]
        return list_scatter_array, list_scatter_cmap
            
    def plot_UMAP(self, list_scatter_array, list_scatter_cmap, list_label=['Parameter $\Phi$',"Spherical interpolation", "Linear interpolation"], 
                save=False, start_end_point = None,
                cmap_list    = ['autumn',"summer","winter"],
                marker_list  = [".","^","v"],
                size_list    = [5,5,5],
                z_order_list = [3,2,1],
                alpha_list   = [1,0.5,0.5],
                ):
        plt.figure(figsize=(8,4))
        #plt.title("UMAP latent representation")
        num_array = len(list_scatter_array)
        num_cmap = 0#len(list_scatter_cmap)
        assert len(list_label) == num_array, "label list hast to be the same length as the list of arrays"
        # twilight_shifted
        for i in range(num_array):
            array = list_scatter_array[i]
            cur_c_map = cmap_list[i]
            if 0 == i:
                b = np.linspace(0,1,self.amount_phi)
                r = np.linspace(0,1,self.amount_theta)
                cur_c = []
                for blue in b:
                    for red in r:
                        cur_c.append([red,blue,0])
                print("len(cur_c)",len(cur_c))
            else:
                cur_c = np.linspace(0,1,array.shape[0])
                print("was here")
                
            cur_marker  = marker_list[i]
            cur_size    = size_list[i]
            cur_z_order = z_order_list[i]
            cur_alpha   = alpha_list[i]

            if start_end_point!=None:# and i == 0:
                if start_end_point >= i:
                    if i == 0:    
                        lable_1 = "first data point"
                        lable_2 = "last data point"
                        plt.scatter(array[0,0],array[0,1], marker="x", c="r", zorder=1000, label=lable_1)
                        plt.scatter(array[-1,0],array[-1,1], marker="x", c="b", zorder=1000, label=lable_2)
                        plt.legend()
                    else:
                        lable_1 = None
                        lable_2 = None
            
            if i ==0:
                plt.scatter(array[:,0].flatten(),array[:,1].flatten(), s=cur_size, marker=cur_marker, c=cur_c, zorder=cur_z_order, alpha=cur_alpha)
            else:
                plt.scatter(array[:,0].flatten(),array[:,1].flatten(), s=cur_size, marker=cur_marker, c=cur_c, cmap=cur_c_map, zorder=cur_z_order, alpha=cur_alpha)
                # plot color bar        
                cbar = plt.colorbar()
                cbar.set_label(list_label[i])

        plt.tight_layout()
        if save:
            name = "amount_phi_" + str(self.amount_phi) + "_delta_phi_" + str(self.delta_phi) + "_"
            plt.savefig(self.results_path + name + "UAMP.png" , dpi=300, bbox_inches='tight')
            print("figure saved")
            # save txt file with difference numbers between interpolations
            file_write_diff = open(self.results_path + name + "diff_art_data.txt","w") 
            L = ["This is Delhi \n","This is Paris \n","This is London \n"]  
            strings = ["MSE of linear interpolation and phi sequence = " + str(self.mse_lin_interpol) + "\n",
                       "MSE of spherical interpolation and phi sequence = " + str(self.mse_sph_interpol) + "\n",
                       "MSE between spherical and linear interpolation = " + str(self.mse_lin_to_sph) + "\n"]
            for s in strings:
                file_write_diff.write(s)
            file_write_diff.close()
        plt.show()




if __name__ == "__main__":
    name_no_metric = "2020-04-28T00-17-24_gan_phi_counter_part_NO_metric_small_lr_ed_lr_07_dlr_samp_ba32_acc065"
    name_metric = "2020-04-27T11-52-58_gan_phi_metric_small_lr_ed_lr_07_dlr_samp_ba32_acc065"
    final_metric = "2020-04-30T00-12-19_gan_phi_first_final_metric"
    final_no_metric = "2020-04-30T00-12-42_gan_phi_first_final_no_metric"

    big_metric_sample = "2020-05-03T23-08-16_gan_big_phi_theta_scale_with_metric_sample"
    big_sample = "2020-05-03T23-06-54_gan_big_phi_theta_scale_without_metric_with_sample"
    big_ = "2020-05-03T23-05-06_gan_big_phi_theta_scale_without_metric"
    big_metric = "2020-05-03T23-02-51_gan_big_phi_theta_scale_with_metric"

    cur_name = big_

    clas_umap = UMAP_(cur_name)
    data_latent, interpol_lin, interpol_sph = clas_umap.create_interpol_images(amount_phi = 100, delta_phi = 15, phi_offset=10, save = False, plot = True)#, delta_phi=90, phi_offset=10, save=False, plot=False)