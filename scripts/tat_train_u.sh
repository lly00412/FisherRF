export CUDA_VISIBLE_DEVICES=0

DATASET_PATH=/mnt/Data2/nerf_datasets/tandt_db/tandt
EXP_PATH=./output/tandt_db_train_u
OBJ=train

python train_w_uncertainty.py -s $DATASET_PATH/$OBJ -m $EXP_PATH/$OBJ