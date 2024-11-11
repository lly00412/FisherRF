export CUDA_VISIBLE_DEVICES=1

DATASET_PATH=/mnt/Data2/nerf_datasets/tandt_db/tandt/
EXP_PATH=./output/tandt/

emsemble_seeds=(0 500 1000 2000 600)
scenes=(train truck)

#DATASET_PATH=/mnt/Data2/nerf_datasets/tandt_db/db/
#EXP_PATH=./output/tandt/

#emsemble_seeds=(0 500 1000 2000 600)
#scenes=(drjohnson playroom)



for OBJ in ${scenes[@]}
do
   python active_train.py -s ${DATASET_PATH}/${OBJ} -m ${EXP_PATH}/${OBJ}/v20/ --train_idxs ${OBJ} --eval \
            --method=H_reg --seed=0 --schema vk --n_inits 30 \
            --iterations 20000 --save_iterations 2000 5000 10000 15000 20000 \
            --test_iterations 2000 5000 10000 15000 20000 \

  for eseed in ${emsemble_seeds[@]}
  do
    python active_train.py -s ${DATASET_PATH}/${OBJ} -m ${EXP_PATH}/${OBJ}/${eseed} --train_idxs ${OBJ} --eval \
            --method=H_reg --seed=0 --schema vk --n_inits 30 \
            --iterations 20000 --save_iterations 2000 5000 10000 15000 20000 \
            --test_iterations 2000 5000 10000 15000 20000 \
            --n_emsemble 20 \
            --emsemble_seed ${eseed}
  done
#    python render_uncertainty_w_emsemble.py -m ${EXP_PATH}/${OBJ}/ \
#          --emsemble_seeds 0 500 1000 2000 600

done