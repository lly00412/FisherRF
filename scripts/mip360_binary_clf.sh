#!/bin/bash

export ROOT_DIR=/mnt/Data2/liyan/FisherRF/output/
export CUDA_VISIBLE_DEVICES=0

#scenes=(kitchen garden bicycle counter bonsai flowers room stump)
scenes=(kitchen)
dataset=m360
#timesteps=(20250822_182134 20250823_022647 20250823_102901 20250823_182949 20250824_022941 20250824_102820 20250824_183135 20250825_023450 20250825_104204 20250825_184925)
#timesteps=(20250822_182134 20250823_022647 20250823_102901 20250823_182949 20250824_022941)
timesteps=(20250822_182134)
metric=psnr

RUN_TIME=$(date +%Y%m%d_%H%M%S)

DATA_DIR=${ROOT_DIR}/${dataset}_data

python binary_classifer_on_view_selection.py \
    --data_dir ${DATA_DIR} \
    --dataset_name ${dataset} \
    --data_timesteps "${timesteps[@]}" \
    --scenes "${scenes[@]}" \
    --target ${metric} \
    --exp_name mlp_${RUN_TIME}_${metric} \
    --loss ce \
    --num_epochs 300 --batch_size 64 --lr 1e-4 \
    --seed 0
