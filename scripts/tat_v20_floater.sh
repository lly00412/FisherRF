DATASET_PATH=/mnt/Data2/nerf_datasets/tandt_db/tandt/
EXP_PATH=./output/tandt/
scenes=(train)

for OBJ in ${scenes[@]}
do
  python render_uncertainty_w_Vcams_floater.py -m ${EXP_PATH}/${OBJ}/v20/ \
          --render_vcam --n_vcam 6 --seed=0 --r_scale 0.1 --test_idxs 15
done