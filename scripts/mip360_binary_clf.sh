#!/bin/bash

export ROOT_DIR=/mnt/Data2/liyan/FisherRF/output/
export CUDA_VISIBLE_DEVICES=0

#scenes=(kitchen garden bicycle counter bonsai flowers room stump)
scenes=(kitchen)
dataset=m360
timestep=20250815_012312
metric=psnr

for SCENE in ${scenes[@]}
do
echo ${SCENE}

DATA_FILE=${ROOT_DIR}/${dataset}_mlp_${timestep}/${SCENE}/candidates.csv

python binary_classifer_on_view_selection.py \
    --data_file ${DATA_FILE} \
    --dataset_name ${dataset} \
    --scene ${SCENE} \
    --target ${metric} \
    --exp_name ${SCENE}_${timestep}_${metric} \
    --loss ce \
    --num_epochs 300 --batch_size 128 --lr 1e-3 \
    --seed 0

done