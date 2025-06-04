#!/bin/bash


SCENES=(lego ship chair drums)
for SCENE in "${SCENES[@]}"
do
	### Random
	DATASET_PATH="/home/liyan/data/Synthetic_NeRF/$SCENE/"
	EXP_PATH="/home/liyan/Results/Synthetic_NeRF/Random/$SCENE/"
	if [ ! -d ${EXP_PATH} ]; then
		mkdir -p ${EXP_PATH};
	fi
	touch  ${EXP_PATH}/log.txt
        echo python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=rand --seed=0 --schema v20seq1_inplace --iterations 30000 --log_every_image  --white_background
	python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=rand --seed=0 --schema v20seq1_inplace --iterations 30000 --log_every_image  --white_background 2>&1 | tee -a ${EXP_PATH}/log.txt



	### Farthest
	DATASET_PATH="/home/liyan/data/Synthetic_NeRF/$SCENE/"
	EXP_PATH="/home/liyan/Results/Synthetic_NeRF/Farthest/$SCENE/"
	if [ ! -d ${EXP_PATH} ]; then
		mkdir -p ${EXP_PATH};
	fi
	touch  ${EXP_PATH}/log.txt
        echo python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=Farthest --seed=0 --schema v20seq1_inplace --iterations 30000 --log_every_image  --white_background
	python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=Farthest --seed=0 --schema v20seq1_inplace --iterations 30000 --log_every_image  --white_background 2>&1 | tee -a ${EXP_PATH}/log.txt



	### FisherRF
	DATASET_PATH="/home/liyan/data/Synthetic_NeRF/$SCENE/"
	EXP_PATH="/home/liyan/Results/Synthetic_NeRF/FisherRF/$SCENE/"
	if [ ! -d ${EXP_PATH} ]; then
		mkdir -p ${EXP_PATH};
	fi
	touch  ${EXP_PATH}/log.txt
        echo python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=H_reg --seed=0 --schema v20seq1_inplace --iterations 30000 --log_every_image  --white_background
	python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=H_reg --seed=0 --schema v20seq1_inplace --iterations 30000 --log_every_image  --white_background 2>&1 | tee -a ${EXP_PATH}/log.txt



	### VCURF
	DATASET_PATH="/home/liyan/data/Synthetic_NeRF/$SCENE/"
	EXP_PATH="/home/liyan/Results/Synthetic_NeRF/vcurf/$SCENE/"
	if [ ! -d ${EXP_PATH} ]; then
		mkdir -p ${EXP_PATH};
	fi
	touch  ${EXP_PATH}/log.txt
        echo python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=vcurf --seed=0 --schema v20seq1_inplace --iterations 30000 --log_every_image  --white_background
	python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=vcurf --seed=0 --schema v20seq1_inplace --iterations 30000 --log_every_image  --white_background 2>&1 | tee -a ${EXP_PATH}/log.txt
done

