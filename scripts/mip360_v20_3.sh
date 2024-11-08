export CUDA_VISIBLE_DEVICES=1
DATASET_PATH=/mnt/Data2/nerf_datasets/m360/
EXP_PATH=./output/m360

scenes=(kitchen garden bicycle counter)
#scenes=(bicycle counter)

# run on fisherRF and VCURF
#for OBJ in ${scenes[@]}
#do
#    python active_train.py -s ${DATASET_PATH}/${OBJ} -m ${EXP_PATH}/${OBJ}/v20/ --train_idxs ${OBJ} --eval \
#            --method=H_reg --seed=0 --schema vk --n_inits 20 \
#            --iterations 20000 --save_iterations 2000 5000 10000 15000 20000 \
#            --test_iterations 2000 5000 10000 15000 20000
##    python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH}/${OBJ}/v20/ \
##          --render_vcam --n_vcam 6 --seed=0 --r_scale 0.1
#done

# run on active-nerf
for OBJ in ${scenes[@]}
do
    python active_train.py -s ${DATASET_PATH}/${OBJ} -m ${EXP_PATH}/${OBJ}/activenerf/ --train_idxs ${OBJ} --eval \
            --method=variance --seed=0 --schema vk --n_inits 20 -r 8 \
            --iterations 20000 --save_iterations 2000 5000 10000 15000 20000 \
            --test_iterations 2000 5000 10000 15000 20000
#    python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH}/${OBJ}/v20/ \
#          --render_vcam --n_vcam 6 --seed=0 --r_scale 0.1
done