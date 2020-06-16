
def get_raw_data_path(run_path):
        raw_data_path = run_path + "eval/"
        print("raw_data_path",raw_data_path)
        raw_data_path = raw_data_path + raw_data_path[raw_data_path[:-7].rfind("/")+1:-6] + "/"#os.listdir(raw_data_path)[-1] + "/" 
        print("raw_data_path",raw_data_path)
        logs_list = list(map(int, os.listdir(raw_data_path)))
        logs_list = np.sort(logs_list)
        raw_data_path = raw_data_path + str(logs_list[-1]) + "/model_outputs/labels/" #TODO should be -1
        if len(os.listdir(raw_data_path))>1:
            raw_data_path_list = []
            list_paths = os.listdir(raw_data_path)
            for i in range(len(list_paths)):
                raw_data_path_list.append(raw_data_path + list_paths[i])
            print("found files ", len(list_paths))
            raw_data_path = raw_data_path_list
            print("one raw_data_path",raw_data_path[-1])
            meta_path = raw_data_path[-1][:raw_data_path[-1].rfind("/")]
            meta_path = meta_path[:meta_path.rfind("/")] 
            #meta_path = os.path.join(os.path.dirname(raw_data_path[-1]), '..')
        else:
            print("raw_data_path",raw_data_path)
            raw_data_path = raw_data_path + os.listdir(raw_data_path)[-1] 
            print("raw_data_path",raw_data_path)
            meta_path = os.path.join(os.path.dirname(raw_data_path), '..')
        return meta_path, raw_data_path

def init_needed_parameters(run_name, need_data_out=True):
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

    assert run_name[0] != "/"
    prefix = "GAN/logs/"
    # run_name = "2020-04-24T17-55-31_gan_phi_var_metric_phi_60_alpha_06"
    run_path = prefix + run_name  #/100000/model_outputs"
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
    
def init_callback_file(run_name, GAN = True, delta = None):
    vae, first_run, config, eig_val, eig_vec, figures_path, gif_path, run_name = get_all_modules(run_name, GAN, delta)
    create_all_figures(vae, first_run, config, eig_val, eig_vec, figures_path, gif_path, run_name)
    
def get_all_modules(run_name):
    root, data_out, config = init_needed_parameters(run_name)
    return init_all_modules(root, data_out, config)
    
def init_all_modules(root, data_out, config):
    # move model to gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create all needed paths
    if "/eval/" in root:
        root = root[:root.find("/eval/")]
    if root[-1] == "/":
        root = root[:-1]
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

    if first_run:
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
    else:
        eig_vec = np.load(pca_path + "eig_vec.npy")
        eig_val = np.load(pca_path + "eig_val.npy")
        
    # load Model with latest checkpoint
    gan = GAN(config)
    latest_chkpt_path = get_latest_checkpoint(checkpoint_root = checkpoint_path)
    vae = gan.generator
    vae.load_state_dict(torch.load(latest_chkpt_path)["netG"])
    vae = vae.to(device)
    
    return vae, first_run, config, eig_val, eig_vec, figures_path, gif_path, run_name
        
def create_all_figures(vae, first_run, config, eig_val, eig_vec, figures_path, gif_path, run_name, save_images=True):
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
    view_pc(model=vae, eig_vec=eig_vec, mu=0, delta=delta_,num_pc=num_pc, num_images=9,save=save_images, figures_path = figures_path, config = config)
    save_image_seq(model = vae, config = config, eig_vec = eig_vec, gif_path = gif_path, mu = 0, delta=delta_, pc_num=num_pc, numb_images=100, loop_gif=True)
    if num_pc > 1:
        cur_num = 0
        pc_start = []
        for i in range(num_pc//2):
            pc_start.append(cur_num)
            cur_num += 2
        for i in range(len(pc_start)):
            view_two_added_pc(model=vae, eig_vec=eig_vec, run_name=run_name, config = config, mu = 0, delta = delta_, pc_num_start=pc_start[i], num_images_x=8, num_images_y=8, save=save_images, figures_path=figures_path)




def plot_t_images(images, save =False, name="", path=None):
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

def new_plot(images, save = False, mu = 4, delta=5,num_images=8):
    plt.figure(figsize=(8,8))
    plt.imshow(images)
    plt.hlines(64,0,8*64)
    plt.vlines(range(64,64*8,64),0,127)
    plt.yticks([32,32+64],("direct sample", "pca sampling"))
    plt.xticks(range(32,64*num_images,64),np.around(np.linspace(mu-4,mu+4,num_images),1))
    plt.xlabel("multiplication parameter $\lambda$")
    plt.xlim(0,64*8-1)
    if save:
        path = get_run_results_path()
        plt.savefig(path + "direct_samp_and_pca.png",dpi=300, bbox_inches='tight')
    plt.show()
    
def get_run_results_path():
    a = time.time()
    timeObj = time.localtime(a)
    cur_time ='%d-%d-%d_%d-%d-%d' % (
    timeObj.tm_year, timeObj.tm_mon, timeObj.tm_mday, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec)
    working_res_directory = "/export/home/rhaecker/documents/research-of-latent-representation/VAE/research/notebooks/new_results/sample_latent/"
    results_path = working_res_directory + "/results_" + cur_time + "/"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)
    return results_path



#########################
## coordinate transfer ##
#########################

def cartesian_to_spherical(data_latent):
    def get_sq_sum(vector, n_dim,i):
        sq_sum = 0
        for i in range(n_dim-1,i-1,-1):
            sq_sum += vector[i]**2
        return sq_sum
    
    n_dim = data_latent.shape[1]
    vectors_parameter = []
    for vector in data_latent:
        spher_para = []
        r = np.sqrt(get_sq_sum(vector, n_dim,0))
        spher_para.append(r)
        # calculate angles
        for i in range(n_dim-2):
            x_cur = vector[i]/np.sqrt(get_sq_sum(vector, n_dim, i))
            phi_cur = np.arccos(x_cur)
            spher_para.append(phi_cur)
        phi_n_1 = np.arccos(vector[n_dim-2]/np.sqrt(vector[n_dim-1]**2+vector[n_dim-2]**2)) 
        if vector[n_dim-1]<0:
            phi_n_1 = 2*np.pi - phi_n_1
        spher_para.append(phi_n_1)
        vectors_parameter.append(spher_para)
    spherical_rep = np.asarray(vectors_parameter)
    return spherical_rep # , np.asarray(arccos_input)

def spherical_to_cartesian(data_latent):
    n_dim = data_latent.shape[1]
    vectors_parameter = []
    for vector in data_latent:
        cartesian_para= []
        for i in range(1,n_dim+1):
            x_i = vector[0]
            for j in range(i-1):
                x_i *= np.sin(vector[j+1])
            if i!=n_dim:
                x_i *= np.cos(vector[i])    
            cartesian_para.append(x_i)
        vectors_parameter.append(cartesian_para)
    cartesian_rep = np.asarray(vectors_parameter)
    return cartesian_rep

def mse_difference_of_arrays(data1,data2):
    data1.flatten()
    data2.flatten()
    assert data1.shape == data2.shape
    difference = (data1 - data2)**2
    normed_diff = np.sum(difference)/data1.shape[0]
    return normed_diff



###################
## interpolation ##
###################

def linear_interpolation(sample_1, smaple_2, num):
    return np.linspace(sample_1,smaple_2, num)

def spherical_interpolation(sample_1, sample_2, num):
    samp_sph_1 = cartesian_to_spherical(np.expand_dims(sample_1,0))
    samp_sph_2 = cartesian_to_spherical(np.expand_dims(sample_2,0))
    
    interpol_sph = np.linspace(samp_sph_1[0],samp_sph_2[0], num)
    interpol_car = spherical_to_cartesian(interpol_sph)
    return interpol_car