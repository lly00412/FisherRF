DATASET_PATH=$1
EXP_PATH=$2
OBJ=$3

echo python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --train_idxs ${OBJ} --eval --method=H_reg --seed=0 --schema v10 --iterations 10000  --white_background --save_iterations 2000 5000 10000 --test_iterations 2000 5000 10000
python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --train_idxs ${OBJ} --method=H_reg --seed=0 --schema v10 --iterations 10000  --white_background --save_iterations 2000 5000 10000 --test_iterations 2000 5000 10000