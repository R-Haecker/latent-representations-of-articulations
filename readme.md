# Representation of three dimensional Objects <br/> using Neural Networks
The code provided in this reposoitory was created during my bachelor thesis in the research group Computer Vision at the Heidelberg Collaboratory for Image processing.  
Below you can find a brief description of this work, for all details you should have a look [here](Bachelor_Thesis.pdf).    
## Abstract
> In this thesis we will investigate the mapping from the image space to the latent space using neural networks. We will focus on image data sets, specifically created to display three-dimensional objects with exact labeled articulations. Using a variational autoencoder in combination with a discriminative network, the aim is to extract and investigate information about articulations from images. This enables us to explore and compare the mapping of specific articulation parameters onto the latent space. The main contribution of this work is the comparison between natural interpolations of articulations and different interpolations in the latent space. Furthermore, we investigate how a metric loss improves the model and how a discriminator helps expand the latent space around observations.

## Model
We used an Variational Autoencoder (VAE) and an adversarial Discriminator as our model.  
The architecture used for the Variational Autoencoder is shown in the following figure.
![alt text]((https://github.com/R-Haecker/latent-representations-of-articulations/eval/readme_figures/vae-architecture.png)

### Discriminator Loss
![alt text]((https://github.com/R-Haecker/latent-representations-of-articulations/eval/readme_figures/disciminator_loss.png)

### VAE Loss
![alt text]((https://github.com/R-Haecker/latent-representations-of-articulations/eval/readme_figures/vae_loss.png)


## Data sets
We use data sets created with tools from this [repository](https://github.com/R-Haecker/python_unity_images/).

### 1) Phi Data set
This data set consist of 10,000 samples containing only one articulation of two cuboids while varying the parameter Phi as shown below. Hence, the articulation is horizontally rotated without any other changes throughout the whole data set.
![alt text]((https://github.com/R-Haecker/latent-representations-of-articulations/eval/readme_figures/data/phi_data.png)

### 2) Varied Data set
This data set consists of 500,000 samples displaying between two and four cuboids. To create this diversified data set we vary the articulation parameters Phi Theta and Lambda while allowing different angles between the cuboids but not different scales of cuboids. To introduce even more complexity we will not only vary the articulation parameters but including the appearances and lighting parameters into into the parameter space. This enables different colors, directional lights, up to four spotlights and four point lights with different settings to be randomly chosen in every image. Therefore, creating the following examples.
![alt text]((https://github.com/R-Haecker/latent-representations-of-articulations/eval/readme_figures/data/examples_big_var_phi_theta_scale.png)

## Experiments

### **Updating the Discriminator**
**updating both networks**,  
&nbsp;&nbsp;&nbsp;&nbsp; at every training step  
**inferior network**,  
&nbsp;&nbsp;&nbsp;&nbsp; comparing the discriminator output of the "false" image for the VAE loss and the discriminator loss
**probabilistic inferior network**, 
&nbsp;&nbsp;&nbsp;&nbsp; inferior method but with a randomness introduced for the decision  
**accuracy threshold**,  
&nbsp;&nbsp;&nbsp;&nbsp; calculate the accuracy of the discriminator predictions and update below a given accuracy threshold
**reducing learning rate**,  
&nbsp;&nbsp;&nbsp;&nbsp; improve the model by reducing the learning rate before the end of training

The following table shows the FID scores on the Varied Data set for networks trained with different methods to update the disciminator during training. Lower is better.

| reducing   learning rate | inferior network | probablistic inferior network | both networks | accuracy threshold |
|------------------------|------------------|-------------------------------|---------------|--------------------|
| without                | 50.53            | 36.39                         | 73.51         | 39.23              |
| with                   | 52.57            | 29.14                         | 53.55         | 28.09              |

### **Sampling from latent space**
We use Principle Component Analysis (PCA) on the latent representation of the Phi validation data set to sample from the principle component (pc) with the highest eigenvalue.  
These latent samples create following images.  
![alt text]((https://github.com/R-Haecker/latent-representations-of-articulations/eval/readme_figures/sampling/images_direct_pca.png)  
Furthermore we used an Uniform Manifold Approximation and Projection (UMAP) for dimension reduction to visualize the 128 dimensional latent space in a 2 dimensional plot. In the following figure we compare the pc latent sampling with the representations of the images from the data set.  
![alt text]((https://github.com/R-Haecker/latent-representations-of-articulations/eval/readme_figures/sampling/real_no_metric_samp.png)  

#### Adding Metric Loss
By adding a metric triplet loss we hoped for a better embedding of the parameter phi in the latent space. Redoing the experiments from before results in the following figures.
![alt text]((https://github.com/R-Haecker/latent-representations-of-articulations/eval/readme_figures/sampling/images_with_metric_direct_pca.png)
![alt text]((https://github.com/R-Haecker/latent-representations-of-articulations/eval/readme_figures/sampling/real_metric_samp.png)

### **Interpolaion in latent space**
We use an image sequence with a natural interpolation of the articulation parameter to create the input images in the following figure. The reconstructed output from the VAE model is displayed as the output.  
![alt text]((https://github.com/R-Haecker/latent-representations-of-articulations/eval/readme_figures/interpol/in_output_img.png)
We will use the latent representation of the first and last image to interpolate between them in the latent space. We use an euclidean interpolation and a spherical interpolation to create the following images.  
![alt text]((https://github.com/R-Haecker/latent-representations-of-articulations/eval/readme_figures/interpol/lin_interpol_img.png)  
![alt text]((https://github.com/R-Haecker/latent-representations-of-articulations/eval/readme_figures/interpol/sph_interpol_img.png)  
We use again a UMAP dimension reduction to compare the two interpolations.
![alt text]((https://github.com/R-Haecker/latent-representations-of-articulations/eval/readme_figures/interpol/umap_right.png)

#### Adding Metric Loss
![alt text]((https://github.com/R-Haecker/latent-representations-of-articulations/eval/readme_figures/interpol/umap_with_metric.png)

## Conclusion
**Updating the Discriminator**  
&nbsp;&nbsp;&nbsp;&nbsp; Accuracy threshold as criteria for updating the discriminator  
&nbsp;&nbsp;&nbsp;&nbsp; Updating the inferior network with additional randomness  

**PCA sampling in the latent space**  
&nbsp;&nbsp;&nbsp;&nbsp; linear PC no representation of an articulation parameter  
&nbsp;&nbsp;&nbsp;&nbsp; metric loss does not improve correlation  

**Interpolation in latent space**  
&nbsp;&nbsp;&nbsp;&nbsp; linear interpolation better than a spherical interpolation  
&nbsp;&nbsp;&nbsp;&nbsp; metric loss decreases the correlation