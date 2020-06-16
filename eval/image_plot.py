import numpy as np
import os
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import matplotlib

def plot_images(dicts, images_per_row = 4, save_fig = True, path = ""):
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
    # Stack all rows vertically together
    #matplotlib.use('TkAgg')
    print("final_img.shape: " + str(final_img.shape))
    plt.figure(figsize=(30,5))
    plt.imshow(final_img)
    #plt.axis('off')
    plt.yticks()
    #k_s=["3","3","4","4","5","5","6","6"]
    stride = [1,2,4,8,16,3,2,3]
    ticks = []
    for i in range(8):
        ticks.append("kernel \n size " + k_s[i] + ",\n stride " + str(stride[i]))
        ticks.append("(" + k_s[i] + ", " + str(stride[i]) + ")")
    plt.xticks(range(32,32+64*9,64), ticks)
    # Plot lines between images to better see the borders of the images
    for i in range(1,numb_y):
        plt.axhline(y = dicts[0]["image"].shape[0]*i-0.7, color="k")
    for i in range(1,numb_x):
        plt.axvline(x = dicts[0]["image"].shape[1]*i-1, color="k")
    # Plot the index of the images onto the images
    plt.xlim(0, (images_per_row)*dicts[0]["image"].shape[1]-1)
    plt.ylim(0, (numb_y)*dicts[0]["image"].shape[0]-1)
    # Save figure if wanted.
    if save_fig:
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + "/final_latent_dim.png", bbox_inches='tight', dpi=300)
    plt.show()

def load_image(root):
    #list_=["1","2","4","8","16","32","64","128","256"]
    llist = []
    #which_image = [0,0,0,1,2,5,0,0,4]
    #for lame in ["inputs/","outputs/"]:
    #lame = "outputs/"
    for i in range(8):
        dictionary = {}
        name = list_[i]
        image_path = os.path.join(root,"image_index_" + str(i) + ".png")
        img = np.asarray(Image.fromarray(io.imread(image_path)))
        '''if which_image[i]>=:
            y_shift = 1
        else:
            y_shift=0
        x_shift = (which_image[i]%3)*64
        picked_img = img[x_shift:64+x_shift,y_shift:64+y_shift,:]
        crop = 0
        y_off= 0
        '''#dictionary["image"] = picked_img[crop-y_off:-y_off-crop,crop:-crop,:]
        dictionary["image"] = picked_img
        llist.append(dictionary)
    return llist

def old_load_image(root):
    list_=["1","2","4","8","16","32","64","128","256"]
    llist = []
    which_image = [0,0,0,1,2,5,0,0,10]
    #for lame in ["inputs/","outputs/"]:
    lame = "outputs/"
    for i in range(len(list_)):
        dictionary = {}
        name = list_[i]
        image_path = os.path.join(root, lame + name + ".png")
        img = np.asarray(Image.fromarray(io.imread(image_path)))
        if i ==8:
            y_shift=1
            x_shift = (0)*64
            picked_img = img[x_shift:64+x_shift,y_shift:64+y_shift,:]
        else:
            if which_image[i]>=2:
                y_shift = 1
            else:
                y_shift=0
            x_shift = (which_image[i]%2)*64
            picked_img = img[x_shift:64+x_shift,y_shift:64+y_shift,:]
            crop = 10
            y_off= 10
            print("which_image[i]",which_image[i])
        dictionary["image"] = picked_img[crop-y_off:-y_off-crop,crop:-crop,:]
        #dictionary["image"] = picked_img
        llist.append(dictionary)
    return llist

if __name__ == "__main__":
    updating_run_names = ["2020-05-02T14-05-36_gan_big_data_phi_theta_scale_no_red_lr_lr_0001_test_updating_3","2020-05-02T14-05-26_gan_big_data_phi_theta_scale_no_red_lr_lr_0001_test_updating_2","2020-05-02T14-05-16_gan_big_data_phi_theta_scale_no_red_lr_lr_0001_test_updating_1","2020-05-02T14-05-06_gan_big_data_phi_theta_scale_no_red_lr_lr_0001_test_updating_0",
                     "2020-05-02T13-59-42_gan_big_data_phi_theta_scale__red_lr_07_lr_0001_test_updating_3","2020-05-02T13-59-32_gan_big_data_phi_theta_scale__red_lr_07_lr_0001_test_updating_2","2020-05-02T13-59-22_gan_big_data_phi_theta_scale__red_lr_07_lr_0001_test_updating_1","2020-05-02T13-59-12_gan_big_data_phi_theta_scale__red_lr_07_lr_0001_test_updating_0",
                     "2020-04-27T17-14-13_gan_phi_basic_red_lr_test_updating_one","2020-04-27T17-13-08_gan_phi_basic_red_lr_test_updating_one_prob","2020-04-27T17-12-01_gan_phi_basic_with_red_lr_test_updating_both","2020-04-27T17-08-10_gan_phi_basic_with_red_lr_testing_updating_acc",
                     "2020-04-27T17-05-14_gan_phi_basic_test_updating_accuracy_065","2020-04-27T17-00-54_gan_phi_basic_test_updating_both","2020-04-27T16-59-40_gan_phi_basic_test_updating_one_prob","2020-04-27T16-57-44_gan_phi_basic_updating_one"]
    list_ = load_image(path)
    plot_images(dicts=list_, images_per_row = 9, save_fig = True, path = path)