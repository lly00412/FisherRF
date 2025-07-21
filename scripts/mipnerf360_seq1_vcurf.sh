export CUDA_VISIBLE_DEVICES=1

DATASET_PATH=/mnt/Data2/nerf_datasets/m360/
EXP_PATH=./output/m360_r_0.5

scenes=(kitchen garden bicycle counter bonsai flowers room stump)

for OBJ in ${scenes[@]}
do

echo python active_train.py -s ${DATASET_PATH}/${OBJ} -m ${EXP_PATH}/${OBJ} --eval --method=vcurf --seed=0 --schema v20seq1_inplace --iterations 20000
python active_train.py -s ${DATASET_PATH}/${OBJ} -m ${EXP_PATH}/${OBJ} --eval --method=vcurf --seed=0 --schema v20seq1_inplace --iterations 20000 \
       --n_vcam=6 --r_scale=0.1

done