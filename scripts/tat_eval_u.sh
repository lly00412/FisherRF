export CUDA_VISIBLE_DEVICES=0

DATASET_PATH=/mnt/Data2/nerf_datasets/tandt_db/tandt
EXP_PATH=./output/tandt_db_train_u
OBJ=train

# Quick check on the current model (trained without --eval, so no test set):
# every 8th training view. These views were used in training, so this only checks
# that the pipeline runs and the maps look sensible; it is not a held-out evaluation.
python render_uncertainty_w_u.py -s $DATASET_PATH/$OBJ -m $EXP_PATH/$OBJ \
    --split train --train_every 8 \
    --vcurf_n_vcam 6 --vcurf_r_scale 0.1 --seed 0

# After retraining with --eval, evaluate on the held-out test set:
# python render_uncertainty_w_u.py -s $DATASET_PATH/$OBJ -m $EXP_PATH/$OBJ --eval \
#     --split test --vcurf_n_vcam 6 --vcurf_r_scale 0.1 --seed 0
