#!/bin/bash

latent_dimensions=(8 16 32 64 128 256)
len=${#latent_dimensions[@]}
for (( i=3; i<$len; i=i+3 ))
do 
    edflow -n var_lin__vae_only_4_conv_blocks_$i -b ./run_bash/vae/vae_only_4_conv_blocks.yaml --linear/latent_dim ${latent_dimensions[$i]} -t &&
    edflow -n var_lin__vae_only_4_conv_blocks_$((i+1)) -b ./run_bash/vae/vae_only_4_conv_blocks.yaml --linear/latent_dim ${latent_dimensions[$((i+1))]} -t &&
    edflow -n var_lin__vae_only_4_conv_blocks_$((i+2)) -b ./run_bash/vae/vae_only_4_conv_blocks.yaml --linear/latent_dim ${latent_dimensions[$((i+2))]} -t
    #echo i_1:${i} latent:${latent_dimensions[${i}]} &&
    #echo i_2:$((i+1)) latent:${latent_dimensions[$((i+1))]}
done