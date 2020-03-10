#!/bin/bash

kernel_size=(1 3 5 2 4)
stride=(1 1 1 2 4)
len=${#kernel_size[@]}
for (( i=3; i<$len; i++ ))
do 
    edflow -n vae__var_conv__vae_3_conv_blocks_up_nearest_$i -b ./run_bash/vae/vae_3_conv_blocks_up_nearest.yaml --conv/kernel_size ${kernel_size[$i]} --conv/stride ${stride[$i]} -t
done