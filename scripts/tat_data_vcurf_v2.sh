export CUDA_VISIBLE_DEVICES=1

#EXP_PATH=./output/tandt/
##seeds=(0 100 500 1000 1500 2000 2500 3000 3500 4000)
#seeds=(2000 2500 3000 3500 4000)
#
#for SEED in ${seeds[@]}
#do
#  # Define run-time once
#  RUN_TIME=$(date +%Y%m%d_%H%M%S)
#
#  # Optionally create a subfolder with this version
#  EXP_PATH_WITH_TIME=${EXP_PATH}_${RUN_TIME}
#
#  # Tank and Temples
#  DATASET_PATH=/mnt/Data2/nerf_datasets/tandt_db/tandt/
#  scenes=(train truck)
#
#  for OBJ in ${scenes[@]}
#    do
#        SCENE_PATH=${DATASET_PATH}/${OBJ}
#        MODEL_PATH=${EXP_PATH_WITH_TIME}/${OBJ}
#
#        python train_uncertainty_clf.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=vcurf --seed=${SEED} --schema v5seq1_inplace \
#               --iterations 1000 --n_vcam=8 --r_scale=0.3 --run_time ${RUN_TIME}
#
#    done
#
#  # Deep Blender
#  DATASET_PATH=/mnt/Data2/nerf_datasets/tandt_db/db/
#  scenes=(drjohnson playroom)
#
#  for OBJ in ${scenes[@]}
#    do
#        SCENE_PATH=${DATASET_PATH}/${OBJ}
#        MODEL_PATH=${EXP_PATH_WITH_TIME}/${OBJ}
#
#        python train_uncertainty_clf.py -s ${SCENE_PATH} -m ${MODEL_PATH} --eval --method=vcurf --seed=${SEED} --schema v5seq1_inplace \
#               --iterations 1000 --n_vcam=8 --r_scale=0.3 --run_time ${RUN_TIME}
#
#    done
#
#done

########### train mlp ##################

export ROOT_DIR=/mnt/Data2/liyan/FisherRF/output/
export CUDA_VISIBLE_DEVICES=0

scenes=(train truck)
dataset=tant_db
timesteps=(20250903_002320 20250903_002540 20250903_031107 20250903_031714 20250903_055722)
metric=psnr

RUN_TIME=$(date +%Y%m%d_%H%M%S)

DATA_DIR=${ROOT_DIR}/${dataset}_data

for OBJ in ${scenes[@]}
do
python binary_classifer_on_view_selection.py \
    --data_dir ${DATA_DIR} \
    --dataset_name ${dataset} \
    --data_timesteps "${timesteps[@]}" \
    --scenes ${OBJ} \
    --target ${metric} \
    --exp_name ${OBJ}_${RUN_TIME}_${metric} \
    --loss ce \
    --num_epochs 300 --batch_size 64 --lr 1e-4 \
    --seed 0
done