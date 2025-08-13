export CUDA_VISIBLE_DEVICES=1

DATASET_PATH=/mnt/Data2/nerf_datasets/m360/
EXP_PATH=./output/m360_mlp

#scenes=(kitchen garden bicycle counter bonsai flowers room stump)
scenes=(garden)

# Define run-time once
RUN_TIME=$(date +%Y%m%d_%H%M%S)

# Optionally create a subfolder with this version
EXP_PATH_WITH_TIME=${EXP_PATH}_${RUN_TIME}

for OBJ in ${scenes[@]}
do
    SCENE_PATH=${DATASET_PATH}/${OBJ}
    MODEL_PATH=${EXP_PATH_WITH_TIME}/${OBJ}

    echo python active_train.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=vcurf --seed=0 --schema v10seq1_inplace --iterations 30000 --run_time ${RUN_TIME}

    python train_uncertainty_clf.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=vcurf --seed=0 --schema v10seq1_inplace \
           --iterations 30000 --n_vcam=8 --r_scale=0.3 --run_time ${RUN_TIME}

done