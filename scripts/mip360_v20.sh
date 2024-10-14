DATASET_PATH=$1
EXP_PATH=$2
OBJ=$3

echo python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --train_idxs ${OBJ} --eval --method=H_reg --seed=0 --schema v20 --iterations 20000 --save_iterations 2000 5000 10000 15000 20000 --test_iterations 2000 5000 10000 15000 20000
python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --train_idxs ${OBJ} --eval \
        --method=H_reg --seed=0 --schema vk --n_inits 10 \
        --iterations 20000 --save_iterations 2000 5000 10000 15000 20000 \
        --test_iterations 2000 5000 10000 15000 20000