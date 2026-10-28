export CUDA_VISIBLE_DEVICES=0

DATASET_PATH=/mnt/Data2/nerf_datasets/tnt_dpgs_gt_data/training_input
EXP_PATH=./output/tnt_sparse
objs=(Barn Caterpillar Ignatius Meetingroom Truck)

for OBJ in ${objs[@]}
do
# --eval holds out every 8th view as the test set for render_uncertainty_w_u.py
python train_w_uncertainty.py -s $DATASET_PATH/$OBJ -m $EXP_PATH/$OBJ --wandb_project gs_uncertainty

#python render_uncertainty_w_u.py -s $DATASET_PATH/$OBJ -m $EXP_PATH/$OBJ \
#    --split test --vcurf_n_vcam 6 --vcurf_r_scale 0.1 --seed 0

done

### rebder GT depth in test set
#B=/mnt/Data2/nerf_datasets/tnt_dpgs_gt_data
#
#for S in Barn Caterpillar Ignatius Meetingroom Truck; do
#  python gt_depthmaps_standalone.py -s $B/GT_data/$S/nvs_test --gt_dir $B/GT_data/$S -r 2 \
#      --split_file $B/GT_data/$S/nvs_test/split_sparse_20.json --split test --out_dir $B/GT_data/$S/nvs_test
#done

####### evaluate nvs test

#B=/mnt/Data2/nerf_datasets/tnt_dpgs_gt_data
#for OBJ in ${objs[@]}
#do
##python render_uncertainty_w_u.py -m $EXP_PATH/$OBJ --split nvs_test \
##    --test_source $B/GT_data/$OBJ/nvs_test
#
#done