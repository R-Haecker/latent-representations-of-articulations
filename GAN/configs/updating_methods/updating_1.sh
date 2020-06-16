#!/bin/bash

up=("one" "one_prob" "both" "accuracy")
ac_th=0.65

i=0
edflow -n gan_big_data_phi_theta_scale_no_red_lr_lr_0001_test_updating_$i -b ./configs/updating_methods/conf.yaml --optimization/update ${up[$i]} -t &
sleep 10;
i=$(( $i + 1 ))
edflow -n gan_big_data_phi_theta_scale_no_red_lr_lr_0001_test_updating_$i -b ./configs/updating_methods/conf.yaml --optimization/update ${up[$i]} -t &
sleep 10;
i=$(( $i + 1 ))
edflow -n gan_big_data_phi_theta_scale_no_red_lr_lr_0001_test_updating_$i -b ./configs/updating_methods/conf.yaml --optimization/update ${up[$i]} -t &
sleep 10;
i=$(( $i + 1 ))
edflow -n gan_big_data_phi_theta_scale_no_red_lr_lr_0001_test_updating_$i -b ./configs/updating_methods/conf.yaml --optimization/update ${up[$i]} --optimization/accuracy_threshold $ac_th -t &