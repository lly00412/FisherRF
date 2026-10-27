export CUDA_VISIBLE_DEVICES=0

DATASET_PATH=/mnt/Data2/nerf_datasets/tandt_db/
EXP_PATH=./output/tandt_db_train_u_eval
objs=(tandt/train tandt/truck db/drjohnson db/playroom)

for OBJ in ${objs[@]}
do

# --eval holds out every 8th view as the test set for render_uncertainty_w_u.py
python train_w_uncertainty.py -s $DATASET_PATH/$OBJ -m $EXP_PATH/$OBJ --eval

python render_uncertainty_w_u.py -s $DATASET_PATH/$OBJ -m $EXP_PATH/$OBJ --eval \
    --split test --vcurf_n_vcam 6 --vcurf_r_scale 0.1 --seed 0

done