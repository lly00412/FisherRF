#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

DATASET_PATH=/home/liyan/data/data/nerf_synthetic/
EXP_PATH=./output/nerf_synthetic_v20

scenes=(ship chair lego drums)

# Define run-time once
RUN_TIME=$(date +%Y%m%d_%H%M%S)

# Optionally create a subfolder with this version
EXP_PATH_WITH_TIME=${EXP_PATH}_${RUN_TIME}

for OBJ in ${scenes[@]}
do
    SCENE_PATH=${DATASET_PATH}/${OBJ}
    MODEL_PATH=${EXP_PATH_WITH_TIME}/${OBJ}

    echo python active_train.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=vcurf --seed=0 --schema v20seq1_inplace --iterations 20000 --run_time ${RUN_TIME}

    python active_train.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=H_reg --seed=0 --schema vk --n_inits 20 \
           --iterations 30000 --run_time ${RUN_TIME} --white_background

done