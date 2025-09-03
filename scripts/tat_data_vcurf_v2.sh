export CUDA_VISIBLE_DEVICES=1

EXP_PATH=./output/tandt/
#seeds=(0 100 500 1000 1500 2000 2500 3000 3500 4000)
seeds=(2000 2500 3000 3500 4000)

for SEED in ${seeds[@]}
do
  # Define run-time once
  RUN_TIME=$(date +%Y%m%d_%H%M%S)

  # Optionally create a subfolder with this version
  EXP_PATH_WITH_TIME=${EXP_PATH}_${RUN_TIME}

  # Tank and Temples
  DATASET_PATH=/mnt/Data2/nerf_datasets/tandt_db/tandt/
  scenes=(train truck)

  for OBJ in ${scenes[@]}
    do
        SCENE_PATH=${DATASET_PATH}/${OBJ}
        MODEL_PATH=${EXP_PATH_WITH_TIME}/${OBJ}

        python train_uncertainty_clf.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=vcurf --seed=${SEED} --schema v5seq1_inplace \
               --iterations 1000 --n_vcam=8 --r_scale=0.3 --run_time ${RUN_TIME}

    done

  # Deep Blender
  DATASET_PATH=/mnt/Data2/nerf_datasets/tandt_db/db/
  scenes=(drjohnson playroom)

  for OBJ in ${scenes[@]}
    do
        SCENE_PATH=${DATASET_PATH}/${OBJ}
        MODEL_PATH=${EXP_PATH_WITH_TIME}/${OBJ}

        python train_uncertainty_clf.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=vcurf --seed=${SEED} --schema v5seq1_inplace \
               --iterations 1000 --n_vcam=8 --r_scale=0.3 --run_time ${RUN_TIME}

    done

done

# Define run-time once
#RUN_TIME=$(date +%Y%m%d_%H%M%S)
#
## Optionally create a subfolder with this version
#EXP_PATH_WITH_TIME=${EXP_PATH}_${RUN_TIME}
#
#for OBJ in ${scenes[@]}
#do
#    SCENE_PATH=${DATASET_PATH}/${OBJ}
#    MODEL_PATH=${EXP_PATH_WITH_TIME}/${OBJ}
#
#    echo python active_train.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=vcurf --seed=${SEED} --schema v5seq1_inplace --iterations 1000 --run_time ${RUN_TIME}
#
#    MLP_CKPT=./ckpts/m360/${OBJ}_psnr/epoch=299.ckpt
#
##    python active_train.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=mlp --seed=0 --schema v20seq1_inplace \
##           --iterations 30000 --n_vcam=8 --r_scale=0.3 --mlp_ckpt=${MLP_CKPT}
#    python active_train.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=mlp --seed=0 --schema v20seq4_inplace \
#           --iterations 20000 --n_vcam=8 --r_scale=0.3 --mlp_ckpt=${MLP_CKPT}
#
#done
