export CUDA_VISIBLE_DEVICES=0

DATASET_PATH=/mnt/Data2/nerf_datasets/m360/
EXP_PATH=./output/m360

emsemble_seeds=(0 500 1000 2000 600)
scenes=(kitchen garden bicycle counter)

for OBJ in ${scenes[@]}
do
#  for eseed in ${emsemble_seeds[@]}
#  do
#    python active_train.py -s ${DATASET_PATH}/${OBJ} -m ${EXP_PATH}/${OBJ}/${eseed} --train_idxs ${OBJ} --eval \
#            --method=H_reg --seed=0 --schema vk --n_inits 20 \
#            --iterations 20000 --save_iterations 2000 5000 10000 15000 20000 \
#            --test_iterations 2000 5000 10000 15000 20000 \
#            --n_emsemble 15 \
#            --emsemble_seed ${eseed}
#  done
    python render_uncertainty_w_emsemble.py -m ${EXP_PATH}/${OBJ}/ \
          --emsemble_seeds 0 500 1000 2000 600

done