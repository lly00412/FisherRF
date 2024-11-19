export CUDA_VISIBLE_DEVICES=0

DATASET_PATH=/mnt/Data2/nerf_datasets/nerf_synthetic/
EXP_PATH=./output/nerf_synthetic
scenes=(ship)

for OBJ in ${scenes[@]}
do

#     python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH}/${OBJ}/v15/ \
#          --render_vcam --n_vcam 2 4 6 8 10 --seed=0 --r_scale 0.1
#
#     python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH}/${OBJ}/v15/ \
#          --render_vcam --n_vcam 6 --seed=0 --r_scale 0.05 0.1 0.3 0.5

     python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH}/${OBJ}/v15/ \
          --render_vcam --n_vcam 6 --seed=0 --r_scale 0.1

done
#
EXP_PATH=./output/m360
scenes=(kitchen)

for OBJ in ${scenes[@]}
do

#     python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH}/${OBJ}/v20/ \
#          --render_vcam --n_vcam 2 4 6 8 10 --seed=0 --r_scale 0.1
#
#     python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH}/${OBJ}/v20/ \
#          --render_vcam --n_vcam 6 --seed=0 --r_scale 0.05 0.1 0.3 0.5
python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH}/${OBJ}/v20/ \
          --render_vcam --n_vcam 6 --seed=0 --r_scale 0.1

done

EXP_PATH=./output/tandt
scenes=(playroom)

for OBJ in ${scenes[@]}
do

#     python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH}/${OBJ}/v20/ \
#          --render_vcam --n_vcam 2 4 6 8 10 --seed=0 --r_scale 0.1
#
#     python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH}/${OBJ}/v20/ \
#          --render_vcam --n_vcam 6 --seed=0 --r_scale 0.05 0.1 0.3 0.5

          python render_uncertainty_w_Vcams_v2.py -m ${EXP_PATH}/${OBJ}/v20/ \
          --render_vcam --n_vcam 6 --seed=0 --r_scale 0.1

done