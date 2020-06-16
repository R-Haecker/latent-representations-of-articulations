#!/bin/bash

dlr=(1 1 2 2)
sample=(False True True False)


i=0
edflow -n gan_var_phi_theta_scale_red_lr_05_lr_0001_test_dlr_sample_$i -b ./configs/big_var_phi_theta/04_27_red_lr_05_lr_0001.yaml --optimization/factor_disc_lr ${dlr[$i]} --optimization/latent_sample ${sample[$i]} -t &
sleep 10;
i=$(( $i + 1 ))
edflow -n gan_var_phi_theta_scale_red_lr_05_lr_0001_test_dlr_sample_$i -b ./configs/big_var_phi_theta/04_27_red_lr_05_lr_0001.yaml --optimization/factor_disc_lr ${dlr[$i]} --optimization/latent_sample ${sample[$i]} -t &
sleep 10;
i=$(( $i + 1 ))
edflow -n gan_var_phi_theta_scale_red_lr_05_lr_0001_test_dlr_sample_$i -b ./configs/big_var_phi_theta/04_27_red_lr_05_lr_0001.yaml --optimization/factor_disc_lr ${dlr[$i]} --optimization/latent_sample ${sample[$i]} -t &
sleep 10;
i=$(( $i + 1 ))
edflow -n gan_var_phi_theta_scale_red_lr_05_lr_0001_test_dlr_sample_$i -b ./configs/big_var_phi_theta/04_27_red_lr_05_lr_0001.yaml --optimization/factor_disc_lr ${dlr[$i]} --optimization/latent_sample ${sample[$i]} -t &