export CUDA_VISIBLE_DEVICES=1

DATASET_PATH=/mnt/Data2/nerf_datasets/nerf_synthetic/
EXP_PATH=./output/nerf_synthetic

emsemble_seeds=(0 500 1000 2000 600)
scenes=(chair lego drums)
#scenes=(ship)

for OBJ in ${scenes[@]}
do
#    python active_train.py -s ${DATASET_PATH}/${OBJ} -m ${EXP_PATH}/${OBJ}/v15/ --train_idxs blender --eval \
#            --method=H_reg --seed=0 --schema vk --n_inits 15 \
#            --iterations 20000 --save_iterations 2000 5000 10000 15000 20000 \
#            --test_iterations 2000 5000 10000 15000 20000
#     python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH}/${OBJ}/v15/ \
#          --render_vcam --n_vcam 6 --seed=0 --r_scale 0.1

# run active-nerf

#    python active_train.py -s ${DATASET_PATH}/${OBJ} -m ${EXP_PATH}/${OBJ}/activenerf/ --train_idxs ${OBJ} --eval \
#            --method=variance --seed=0 --schema vk --n_inits 15 \
#            --iterations 20000 --save_iterations 2000 5000 10000 15000 20000 \
#            --test_iterations 2000 5000 10000 15000 20000
#
#    python render_uncertainty_w_activenerf.py -m ${EXP_PATH}/${OBJ}/activenerf/ \
#          --render_vcam --n_vcam 6 --seed=0 --r_scale 0.1

# train vcurf

python active_vcurf.py -s ${DATASET_PATH}/${OBJ} -m ${EXP_PATH}/${OBJ}/activevcurf3/ --train_idxs ${OBJ} --eval \
            --method=vcam --seed=0 --schema vk --n_inits 15 --n_vcam 6 --r_scale 0.1 \
            --iterations 20000 --save_iterations 2000 5000 10000 15000 20000 \
            --test_iterations 2000 5000 10000 15000 20000 \
            --checkpoint_iterations 100 5000 10000 20000

python render_uncertainty_w_activevcurf.py -m ${EXP_PATH}/${OBJ}/activevcurf3/ \
          --render_vcam --n_vcam 6 --seed=0 --r_scale 0.1

done