export CUDA_VISIBLE_DEVICES=1

emsemble_seeds=(0 500 1000 2000 600)

DATASET_PATH=/mnt/Data2/nerf_datasets/tandt_db/tandt/
EXP_PATH=./output/tandt/
scenes=(train truck)


for OBJ in ${scenes[@]}
do

python active_vcurf2.py -s ${DATASET_PATH}/${OBJ} -m ${EXP_PATH}/${OBJ}/activevcurf4/ --train_idxs ${OBJ} --eval \
            --method=vcam --seed=0 --schema vk --n_inits 20 --n_vcam 6 --r_scale 0.1 \
            --iterations 20000 --save_iterations 2000 5000 10000 15000 20000 \
            --test_iterations 2000 5000 10000 15000 20000 \
            --checkpoint_iterations 100 5000 10000 20000

python render_uncertainty_w_activevcurf.py -m ${EXP_PATH}/${OBJ}/activevcurf4/ \
          --render_vcam --n_vcam 6 --seed=0 --r_scale 0.1
done



DATASET_PATH=/mnt/Data2/nerf_datasets/tandt_db/db/
EXP_PATH=./output/tandt/
scenes=(drjohnson playroom)



for OBJ in ${scenes[@]}
do
python active_vcurf2.py -s ${DATASET_PATH}/${OBJ} -m ${EXP_PATH}/${OBJ}/activevcurf4/ --train_idxs ${OBJ} --eval \
            --method=vcam --seed=0 --schema vk --n_inits 20 --n_vcam 6 --r_scale 0.1 \
            --iterations 20000 --save_iterations 2000 5000 10000 15000 20000 \
            --test_iterations 2000 5000 10000 15000 20000 \
            --checkpoint_iterations 100 5000 10000 20000

python render_uncertainty_w_activevcurf.py -m ${EXP_PATH}/${OBJ}/activevcurf4/ \
          --render_vcam --n_vcam 6 --seed=0 --r_scale 0.1

done