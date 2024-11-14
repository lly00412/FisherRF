export CUDA_VISIBLE_DEVICES=0

DATASET_PATH=/mnt/Data2/nerf_datasets/nerf_synthetic/
EXP_PATH=./output/nerf_synthetic

emsemble_seeds=(0 500 1000 2000 600)
scenes=(lego)
#scenes=(ship)

for OBJ in ${scenes[@]}
do

     python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH}/${OBJ}/v15/ \
          --render_vcam --n_vcam 6 --seed=0 --r_scale 0.1 --test_idxs 125

done