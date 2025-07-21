export CUDA_VISIBLE_DEVICES=0

DATASET_PATH=/mnt/Data2/nerf_datasets/nerf_synthetic/
EXP_PATH=./output/nerf_synthetic

scenes=(ship chair lego drums hotdog ficus materials mic)

for OBJ in ${scenes[@]}
do

echo python active_train.py -s ${DATASET_PATH}/${OBJ} -m ${EXP_PATH}/${OBJ} --eval --method=vcurf --seed=0 --schema v20seq1_inplace --iterations 30000  --white_background
python active_train.py -s ${DATASET_PATH}/${OBJ} -m ${EXP_PATH}/${OBJ} --eval --method=vcurf --seed=0 --schema v20seq1_inplace --iterations 30000  --white_background \
       --n_vcam=6 --r_scale=0.1

done