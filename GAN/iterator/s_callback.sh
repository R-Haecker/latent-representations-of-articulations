#!/bin/bash

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -r|--run_path)
    run_path="$2"
    shift # past argument
    shift # past value
    ;;
    -d|--delta)
    delta_val="$2"
    shift # past argument
    shift # past value
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

root_path="/export/home/rhaecker/documents/research-of-latent-representation/GAN/logs/"
root_path="$root_path$run_path"
first_final_path="/eval/"
root_path="$root_path$first_final_path$run_path"
eval_number=0
for entry in "$root_path"/*
do
    current_number="$(basename $entry)"
    if [ "$current_number" -gt "$eval_number" ] 
    then 
        eval_number="$current_number"
    fi
done
echo "loading the model from step $eval_number"
root_path="$root_path"/"$eval_number"/model_outputs""

edeval -m "$root_path" -c rici:iterator.callback.latent --latent_delta "$delta_val"