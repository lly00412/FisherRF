#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
SCENES=(bicycle kitchen counter garden)

for SCENE in ${SCENES[@]}
do
      EXP_PATH="/home/liyan/data/data/MipNeRF360_pretrained/${SCENE}/v20/"
      python render_uncertainty_w_different_methods.py -m ${EXP_PATH} --scene ${SCENE} --training_views "./mip360_training_views.yaml" --seed 29506 \
      --csv_file "./output/m360_seq1_data/${SCENE}/candidates.csv"
done