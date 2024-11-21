export CUDA_VISIBLE_DEVICES=0

DATASET_PATH=/mnt/Data2/nerf_datasets/nerf_synthetic/

emsemble_seeds=(0 500 1000 2000 600)
EXP_PATH=./output/m360
scenes=(bicycle)
#
for OBJ in ${scenes[@]}
do

     python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH}/${OBJ}/v20/ \
          --render_vcam --n_vcam 6 --seed=0 --r_scale 0.1 --test_idxs 18

done

#EXP_PATH=./output/nerf_synthetic
#scenes=(chair)
#
#for OBJ in ${scenes[@]}
#do
#
#     python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH}/${OBJ}/v15/ \
#          --render_vcam --n_vcam 6 --seed=0 --r_scale 0.1 --test_idxs 53
#
##     python render_uncertainty_w_emsemble.py -m ${EXP_PATH}/${OBJ}/ \
##          --emsemble_seeds 0 500 1000 2000 600 --test_idxs 84
#
#done


#EXP_PATH=./output/tandt
#scenes=(train)
#
#for OBJ in ${scenes[@]}
#do
#
#     python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH}/${OBJ}/v20/ \
#          --render_vcam --n_vcam 6 --seed=0 --r_scale 0.1 --test_idxs 9
#
##     python render_uncertainty_w_emsemble.py -m ${EXP_PATH}/${OBJ}/ \
##          --emsemble_seeds 0 500 1000 2000 600 --test_idxs 84
#
#done