export CUDA_VISIBLE_DEVICES=0

DATASET_PATH=/mnt/Data2/nerf_datasets/tandt_db/
EXP_PATH=./output/tandt_db

# Define run-time once
RUN_TIME=$(date +%Y%m%d_%H%M%S)

# Optionally create a subfolder with this version
EXP_PATH_WITH_TIME=${EXP_PATH}_${RUN_TIME}

tant_scenes=(train truck)
db_scenes=(drjohnson playroom)

for OBJ in ${tant_scenes[@]}
do
    SCENE_PATH=${DATASET_PATH}/tandt/${OBJ}
    MODEL_PATH=${EXP_PATH_WITH_TIME}/${OBJ}

    echo python active_train.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=mlp --seed=${SEED} --schema v20seq1_inplace --iterations 20000 --run_time ${RUN_TIME}

    MLP_CKPT=./ckpts/tant_db/${OBJ}_psnr/epoch=299.ckpt

    python active_train.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=mlp --seed=0 --schema v20seq1_inplace \
           --iterations 20000 --n_vcam=8 --r_scale=0.3 --mlp_ckpt=${MLP_CKPT}

done

for OBJ in ${db_scenes[@]}
do
    SCENE_PATH=${DATASET_PATH}/db/${OBJ}
    MODEL_PATH=${EXP_PATH_WITH_TIME}/${OBJ}

    echo python active_train.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=mlp --seed=${SEED} --schema v20seq1_inplace --iterations 20000 --run_time ${RUN_TIME}

    MLP_CKPT=./ckpts/tant_db/${OBJ}_psnr/epoch=299.ckpt

    python active_train.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=mlp --seed=0 --schema v20seq1_inplace \
           --iterations 20000 --n_vcam=8 --r_scale=0.3 --mlp_ckpt=${MLP_CKPT}

done