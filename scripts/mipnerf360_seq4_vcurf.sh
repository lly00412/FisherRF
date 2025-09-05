export CUDA_VISIBLE_DEVICES=1

DATASET_PATH=/mnt/Data2/nerf_datasets/m360/
EXP_PATH=./output/m360_seq4

scenes=(kitchen garden bicycle counter bonsai flowers room stump)
seeds=(100 500 1000 1500)

#for SEED in ${seeds[@]}
#do
#  # Define run-time once
#  RUN_TIME=$(date +%Y%m%d_%H%M%S)
#
#  # Optionally create a subfolder with this version
#  EXP_PATH_WITH_TIME=${EXP_PATH}_${RUN_TIME}
#
#  for OBJ in ${scenes[@]}
#  do
#      SCENE_PATH=${DATASET_PATH}/${OBJ}
#      MODEL_PATH=${EXP_PATH_WITH_TIME}/${OBJ}
#
#      echo python active_train.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=vcurf --seed=${SEED} --schema v5seq1_inplace --iterations 1000 --run_time ${RUN_TIME}
#
##      python train_uncertainty_clf.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=vcurf --seed=${SEED} --schema v5seq1_inplace \
##             --iterations 1000 --n_vcam=8 --r_scale=0.3 --run_time ${RUN_TIME}
#      MLP_CKPT=
#
#      python active_train.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=mlp --seed=${SEED} --schema v20seq1_inplace \
#             --iterations 30000 --n_vcam=8 --r_scale=0.3
#
#  done
#done
for SEED in ${seeds[@]}
do

# Define run-time once
RUN_TIME=$(date +%Y%m%d_%H%M%S)

# Optionally create a subfolder with this version
EXP_PATH_WITH_TIME=${EXP_PATH}_${RUN_TIME}

for OBJ in ${scenes[@]}
do
    SCENE_PATH=${DATASET_PATH}/${OBJ}
    MODEL_PATH=${EXP_PATH_WITH_TIME}/${OBJ}

    echo python active_train.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=vcurf --seed=${SEED} --schema v20seq4_inplace --iterations 20000 --run_time ${RUN_TIME}

    MLP_CKPT=./ckpts/m360/${OBJ}_psnr/epoch=299.ckpt

#    python active_train.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=mlp --seed=0 --schema v20seq1_inplace \
#           --iterations 30000 --n_vcam=8 --r_scale=0.3 --mlp_ckpt=${MLP_CKPT}
    python active_train.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=mlp --seed=${SEED} --schema v20seq4_inplace \
           --iterations 20000 --n_vcam=8 --r_scale=0.3 --mlp_ckpt=${MLP_CKPT}

done
done