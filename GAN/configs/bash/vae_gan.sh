#!/bin/bash

reconstruction_loss_weight=(10 20 50 100)

len=${#reconstruction_loss_weight[@]}
for (( i=0; i<$len; i=i+1 ))
do 
    edflow -n vae_gan_var_recon_weight_$i -b ./configs/vae_gan.yaml --losses/reconstruction_loss_weight ${reconstruction_loss_weight[$i]} -t
done